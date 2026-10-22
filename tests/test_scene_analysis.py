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


    def test_one_still_schema_forbids_motion_and_binds_evidence_ids(self):
        schema = observation_response_format(['a'*64])['json_schema']['schema']
        ids = schema['properties']['evidence_ids']
        self.assertEqual(ids['items']['enum'], ['a'*64])
        movement = schema['properties']['observations']['properties']['camera_movement']
        self.assertEqual(movement['type'], 'string')
        self.assertEqual(movement['enum'], [''])
        msg = __import__('backend.scene_analysis', fromlist=['observation_messages']).observation_messages([], ['a'*64])[0]['content']
        self.assertIn('camera_movement field MUST be an empty string', msg)
        other = observation_response_format(['a'*64, 'b'*64])['json_schema']['schema']
        self.assertNotIn('enum', other['properties']['observations']['properties']['camera_movement'])

    def test_lm_studio_nonempty_single_still_motion_is_discarded_not_accepted(self):
        for hallucinated_motion in [
            'Camera pans left',
            'Locked-off static camera',
            'A tracking shot follows the actor.',
        ]:
            response=observation()
            response['observations']['camera_movement'] = hallucinated_motion
            response['observations']['framing'] = 'Medium-wide, seated person by a window.'
            result={'choices':[{'message':{'content':json.dumps(response)},'finish_reason':'stop'}]}
            parsed=parse_observation(result,['a'*64])
            self.assertIsNone(parsed['observations']['camera_movement'])
            self.assertEqual(parsed['observations']['framing'],'Medium-wide, seated person by a window.')
            self.assertEqual(parsed['evidence_ids'],['a'*64])
            # Direct non-normalizing validation still blocks unsupported
            # motion, so persistence cannot bypass the safety invariant.
            with self.assertRaisesRegex(ModelError,'camera movement is unsupported'):
                validate_observation(response,['a'*64])

    def test_real_visual_schema_incompatibilities_report_field_not_model_text(self):
        for mutator, expected in [
            (lambda o:o.update(evidence_ids=['b'*64]),'evidence IDs'),
            (lambda o:o['observations'].update(framing='<Picture 4>'),'framing'),
            (lambda o:o.update(confidence=1.5),'confidence'),
            (lambda o:o.update(garbage='extra'), 'top-level schema'),
        ]:
            payload=observation()
            mutator(payload)
            reply={'choices':[{'message':{'content':json.dumps(payload)},'finish_reason':'stop'}]}
            with self.assertRaises(ModelError) as error:
                parse_observation(reply,['a'*64])
            self.assertIn(expected,error.exception.message)
            self.assertNotIn('b'*64,error.exception.message)
            self.assertNotIn('<Picture',error.exception.message)

    def test_multi_image_motion_allowed_only_with_matching_ordered_evidence(self):
        record=observation()
        record['evidence_ids']=['a'*64, 'b'*64]
        record['observations']['camera_movement']='Between the two ordered stills, framing shifts left.'
        result={'choices':[{'message':{'content':json.dumps(record)},'finish_reason':'stop'}]}
        self.assertEqual(parse_observation(result,['a'*64,'b'*64])['observations']['camera_movement'],
                         record['observations']['camera_movement'])
        with self.assertRaisesRegex(ModelError,'evidence IDs'):
            parse_observation(result,['b'*64,'a'*64])

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

    def test_subject_prompt_requests_visible_complexion_without_ancestry_inference(self):
        from backend.scene_analysis import observation_messages
        system=observation_messages([],['a'*64])[0]['content']
        for expected in [
            'visible skin tone/complexion',
            'facial hair',
            'hair texture/color',
            'dark-brown skin',
            'Do not infer a person\'s nationality, ethnic ancestry',
        ]:
            self.assertIn(expected,system)
        item=observation()
        item['subjects']=[
            'Adult man with dark-brown skin, short dark hair and beard, '
            'wearing a light-purple shirt with a patterned tie.'
        ]
        self.assertEqual(validate_observation(item,['a'*64]),item)
