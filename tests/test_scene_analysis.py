import base64
import copy
import io
import json
import unittest
from unittest.mock import patch
from PIL import Image
from backend.scene_analysis import FIELDS, prepare_evidence, parse_observation, validate_observation, observation_response_format
from backend.models.api_provider_backend import ApiProviderBackend, ApiConnection
from backend.models.contract import ModelError


def evidence():
    output=io.BytesIO();Image.new('RGB',(160,96),(70,90,100)).save(output,format='PNG')
    return {'version':1,'images':[{'id':'a'*64,'pts':'67/24','data_url':'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()}]}


def observation():
    return {'version':1,'evidence_ids':['a'*64],'observations':dict.fromkeys(FIELDS),'subjects':['Seated person at frame left.'],'confidence':.7}


class SceneAnalysisTests(unittest.TestCase):
    def test_evidence_inline_images_only(self):
        content,ids,tokens=prepare_evidence(evidence())
        self.assertEqual(ids,['a'*64]);self.assertGreater(tokens,0)
        for images in [[],[{}],evidence()['images']*2,[{'id':'a'*64,'data_url':'http://localhost/image.png'}]]:
            with self.assertRaises(ModelError):prepare_evidence({'version':1,'images':images})
        with self.assertRaises(ModelError):prepare_evidence({**evidence(),'version':2})

    def test_schema_evidence_markup_motion(self):
        value=observation();self.assertEqual(validate_observation(value,['a'*64]),value)
        for mutate in [lambda x:x.update(evidence_ids=['b'*64]),lambda x:x.update(confidence=float('nan')),
                       lambda x:x['observations'].update(camera_movement='pan'),lambda x:x['observations'].update(acting='<d>hello</d>'),
                       lambda x:x.update(dialogue='invented')]:
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(ModelError):validate_observation(bad,['a'*64])

    def test_truncated_invalid_and_interrupted_responses(self):
        result={'choices':[{'message':{'content':json.dumps(observation())},'finish_reason':'stop'}]}
        self.assertEqual(parse_observation(result,['a'*64]),observation())
        result['choices'][0]['finish_reason']='length'
        with self.assertRaises(ModelError):parse_observation(result,['a'*64])
        result['choices'][0]['finish_reason']='stop';result['choices'][0]['message']['content']='{'
        with self.assertRaises(ModelError):parse_observation(result,['a'*64])

    def backend(self,vision=True):
        b=ApiProviderBackend();c=ApiConnection('connection','custom','http://localhost:1234/v1','SECRET',True,16384)
        model=b._model_info(c,{'id':'vision','architecture':{'input_modalities':['text','image'] if vision else ['text']},'context_length':16384})
        c.models=[model];b._connections[c.id]=c
        return b,model

    def test_existing_transport_separate_from_h3_pipeline(self):
        backend,model=self.backend()
        result={'choices':[{'message':{'content':json.dumps(observation())},'finish_reason':'stop'}]}
        with patch.object(backend,'_request_chat_completion_stream',return_value=result) as stream,patch('backend.models.api_provider_backend.run_h3_pipeline',side_effect=AssertionError('ordinary H3 pipeline called')):
            backend.prepare_request();value=backend.analyze_scene(model,evidence())
        self.assertEqual(value['version'],1);self.assertEqual(value['observation'],observation())
        payload=stream.call_args.args[1];self.assertEqual(payload['model'],'vision');self.assertEqual(payload['temperature'],0)
        self.assertNotIn('SECRET',json.dumps(backend.scene_models()))

    def test_vision_context_cancel_and_expired_connection(self):
        b,model=self.backend(False)
        with self.assertRaises(ModelError):b.analyze_scene(model,evidence())
        b,model=self.backend();b.cancel_event.set()
        with self.assertRaises(ModelError):b.analyze_scene(model,evidence())
        b.prepare_request();body=evidence();body['context_budget']=2048
        with self.assertRaises(ModelError):b.analyze_scene(model,body)
        b.disconnect('connection')
        with self.assertRaises(ModelError):b.analyze_scene(model,evidence())

    def test_local_structured_output_and_fenced_reply(self):
        backend,model=self.backend()
        backend._connections['connection'].compatibility_profile='lm_studio'
        result={'choices':[{'message':{'content':'```json\n'+json.dumps(observation())+'\n```'},'finish_reason':'stop'}]}
        with patch.object(backend,'_request_chat_completion_stream',return_value=result) as stream:
            backend.prepare_request()
            record=backend.analyze_scene(model,evidence())
        self.assertEqual(record['observation'],observation())
        payload=stream.call_args.args[1]
        schema=payload['response_format']['json_schema']
        self.assertTrue(schema['strict'])
        properties = schema['schema']['properties']['observations']['properties']
        self.assertEqual(set(properties),FIELDS)
        self.assertTrue(all(spec['type'] == 'string' for spec in properties.values()))
        self.assertFalse(backend.scene_models()[0]['remote'])
        backend._connections['connection'].base_url='https://external.example/v1'
        self.assertTrue(backend.scene_models()[0]['remote'])

    def test_fenced_json_does_not_relax_schema_validation(self):
        bad=observation()
        bad['observations']['acting']='<d>[English] fake dialogue</d>'
        result={'choices':[{'message':{'content':'```json\n'+json.dumps(bad)+'\n```'},'finish_reason':'stop'}]}
        with self.assertRaises(ModelError):parse_observation(result,['a'*64])

    def test_complete_reasoning_prefix_and_partial_reasoning_fail_closed(self):
        good='\x3cthink>Internal visual reasoning.\x3c/think>\n'+json.dumps(observation())
        result={'choices':[{'message':{'content':good},'finish_reason':'stop'}]}
        self.assertEqual(parse_observation(result,['a'*64]),observation())
        result['choices'][0]['message']['content']='\x3cthink>Unfinished\n'+json.dumps(observation())
        with self.assertRaises(ModelError):
            parse_observation(result,['a'*64])
        result['choices'][0]['message']['content']='\x3cthink>Done\x3c/think>\n'+json.dumps({
            **observation(),'confidence':float('nan')})
        with self.assertRaises(ModelError):
            parse_observation(result,['a'*64])


    def test_lm_studio_empty_string_sentinel_normalizes_to_null(self):
        payload = observation()
        payload['observations']['acting'] = ''
        payload['observations']['framing'] = 'Medium shot.'
        message = {'choices':[{'message':{'content': json.dumps(payload)},
                               'finish_reason':'stop'}]}
        parsed = parse_observation(message, ['a'*64])
        self.assertIsNone(parsed['observations']['acting'])
        self.assertEqual(parsed['observations']['framing'], 'Medium shot.')
        # Do not admit spoofed H3 markup through the normalization path.
        payload['observations']['framing'] = '<Picture 1>'
        message['choices'][0]['message']['content'] = json.dumps(payload)
        with self.assertRaises(ModelError):
            parse_observation(message, ['a'*64])
