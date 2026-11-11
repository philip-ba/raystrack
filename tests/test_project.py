"""Project input rejection and real disk selective read/restart contracts."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from raystrack.project import FORMAT, load_project, utc_timestamp
from raystrack.project_store import ResultReader, ResultWriter


def fixture(tmp_path):
    np.save(tmp_path/'vertices.npy', np.array([[0,0,0],[1,0,0],[0,1,0]], np.float32))
    np.save(tmp_path/'faces.npy', np.array([[0,1,2]], np.int32))
    (tmp_path/'sensors.csv').write_text('id,group,x,y,z,nx,ny,nz\na,outdoor,0,0,1,0,0,1\nb,shelter,1,1,1,0,0,1\n')
    spec = dict(schema=FORMAT, version=1, units={'length':'m','irradiance':'W/m2'}, coordinates='ENU',
        prototypes=[dict(id='p',vertices='vertices.npy',faces='faces.npy')],
        instances=[dict(id='one',prototype='p',material='black'),dict(id='two',prototype='p',material='black')],
        materials={'black':dict(type='opaque',front=0)}, sensors='sensors.csv',
        site=dict(latitude_deg=52.52,longitude_deg=13.405,utc_offset_minutes=60),
        weather=dict(type='synthetic',year=2025,start_utc='2025-06-21T10:00:00Z',interval_minutes=15,steps=4),
        states=[dict(timestamp_utc='2025-06-21T10:15:00Z', transforms={'one':np.eye(4).tolist()})])
    path = tmp_path/'project.json'
    path.write_text(json.dumps(spec))
    return path, spec


def test_loader_prototypes_weather_and_hold_state(tmp_path):
    path, _ = fixture(tmp_path)
    project = load_project(path)
    assert project.scene['one'].mesh is project.scene['two'].mesh
    assert project.timeline.timestamps[0] == np.datetime64('2025-06-21T10:07:30')
    np.testing.assert_array_equal(project.timeline.durations_hours, [.25]*4)
    assert np.all(project.dni > 0)
    assert project.sensor_ids == ('a','b')
    assert project.states[0]['timestamp_utc'] == np.datetime64('2025-06-21T10:15:00')
    assert set(project.source_hashes) == {'project.json','vertices.npy','faces.npy','sensors.csv'}


def test_pose_csv_merges_inline_edits_and_captures_source_hash(tmp_path):
    path,spec=fixture(tmp_path)
    spec['state_tracks']={'path':'poses.csv','interpolation':'hold'}
    (tmp_path/'poses.csv').write_text('timestamp_utc,instance_id,x,y,z,yaw_deg,material\n'
        '2025-06-21T10:15:00Z,two,2,3,4,90,black\n')
    path.write_text(json.dumps(spec))
    project=load_project(path)
    assert len(project.states)==1
    np.testing.assert_allclose(project.states[0]['transforms']['two'],
        [[0,-1,0,2],[1,0,0,3],[0,0,1,4],[0,0,0,1]],atol=1e-14)
    assert project.states[0]['materials']=={'two':'black'}
    assert 'one' in project.states[0]['transforms']
    assert 'poses.csv' in project.source_hashes
    assert 'two' not in project.manifest['states'][0]['transforms']


@pytest.mark.parametrize('rows,match',[
    ('2025-06-21T10:15:00Z,bad,0,0,0,0,\n','unknown'),
    ('2025-06-21T10:07:30Z,two,0,0,0,0,\n','boundary'),
    ('2025-06-21T10:15:00Z,one,0,0,0,0,\n','duplicate'),
    ('2025-06-21T10:15:00Z,two,0,0,0,nan,\n','finite'),
    ('2025-06-21T10:15:00Z,two,0,,,0,\n','requires all'),
    ('2025-06-21T10:15:00Z,two,,,,,\n','requires a pose'),
    ('2025-06-21T10:15:00Z,two,0,0,0,0,\n2025-06-21T10:15:00Z,two,1,0,0,0,\n','duplicate'),
    ('2025-06-21T10:30:00Z,two,0,0,0,0,\n2025-06-21T10:15:00Z,two,1,0,0,0,\n','decrease'),
])
def test_pose_csv_rejects_ambiguous_or_invalid_tracks(tmp_path,rows,match):
    path,spec=fixture(tmp_path)
    spec['state_tracks']={'path':'poses.csv'}
    (tmp_path/'poses.csv').write_text('timestamp_utc,instance_id,x,y,z,yaw_deg,material\n'+rows)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match=match):
        load_project(path)


@pytest.mark.parametrize('mutation,match', [
    (lambda s:s.update(version=2),'schema'),
    (lambda s:s.update(units={'length':'cm','irradiance':'W/m2'}),'units'),
    (lambda s:s.update(coordinates='NED'),'coordinates'),
    (lambda s:s['instances'][0].update(material='unknown'),'unknown'),
    (lambda s:s['instances'][1].update(id='one'),'duplicate'),
    (lambda s:s['prototypes'][0].update(vertices='../escape.npy'),'escapes'),
    (lambda s:s['prototypes'][0].update(vertices='missing.npy'),'missing'),
    (lambda s:s['states'][0].update(timestamp_utc='2025-06-21T10:15:00'),'UTC'),
    (lambda s:s['states'][0].update(timestamp_utc='2025-06-21T10:16:00Z'),'boundary'),
    (lambda s:s['states'][0].update(interpolation='linear'),'hold'),
    (lambda s:s['states'][0].update(transforms={'bad':np.eye(4).tolist()}),'unknown'),
])
def test_bad_project_inputs(tmp_path,mutation,match):
    path,spec=fixture(tmp_path)
    mutation(spec)
    path.write_text(json.dumps(spec))
    with pytest.raises((ValueError,FileNotFoundError),match=match):
        load_project(path)


def writer(path,**kwargs):
    return ResultWriter(path,configuration={'source':'abc','seed':7},sensor_ids=['a','b','c'],
                        components=['direct','sky'],bands=['solar'],**kwargs)


def append(w,start,count=2,revision=0):
    timestamps=np.datetime64('2025-01-01T00:30','us')+np.arange(start,start+count)*np.timedelta64(1,'h')
    values=np.arange(start*6,(start+count)*6,dtype=float).reshape(count,3,2,1)
    durations=np.array([.5,1.5][:count])
    w.append(timestamps,durations,values,np.ones_like(values)*.1,revision,revision)
    return values,durations


@pytest.mark.parametrize('storage',['npy','npz'])
def test_selective_reads_resume_duration_exposure_and_csv(tmp_path,storage):
    w=writer(tmp_path,storage=storage)
    first,durations=append(w,0)
    resumed=writer(tmp_path,storage=storage,resume=True)
    assert resumed.completed_steps==2
    second,_=append(resumed,2,revision=1)
    manifest=resumed.finish()
    expected=((first+second)*durations[:,None,None,None]).sum(axis=0)
    np.testing.assert_array_equal(manifest['exposure_wh_m2'],expected)
    reader=ResultReader(tmp_path)
    selected=reader.read(2,3,sensor_ids=['c'],components=['sky'])
    np.testing.assert_array_equal(selected['values'],second[:1,2:3,1:2,:])
    assert len(reader.files_read)==(4 if storage=='npy' else 1)
    assert all('block-000001-' in name for name in reader.files_read)
    times=reader.read_times('2025-01-01T02:00:00Z','2025-01-01T03:00:00Z',sensor_ids=['a'])
    assert len(times['timestamps'])==1
    reader.export_csv(tmp_path/'selected.csv',2,3,sensor_ids=['c'],components=['sky'])
    rows=list(csv.DictReader((tmp_path/'selected.csv').open()))
    assert len(rows)==1 and rows[0]['sensor_id']=='c'
    assert rows[0]['timestamp_utc'].endswith('Z')
    with pytest.raises(ValueError,match='append'):
        append(resumed,4)


def test_store_tampering_resume_config_and_path_containment(tmp_path):
    w=writer(tmp_path)
    append(w,0)
    with pytest.raises(ValueError,match='configuration'):
        ResultWriter(tmp_path,configuration={},sensor_ids=['a','b','c'],components=['direct','sky'],bands=['solar'],resume=True)
    target=tmp_path/w.manifest['blocks'][0]['files']['values']['path']
    target.write_bytes(target.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='checksum'):
        ResultReader(tmp_path).read()
    with pytest.raises(ValueError,match='checksum'):
        writer(tmp_path,resume=True)
    manifest=w.manifest
    manifest['blocks'][0]['files']['values']['path']='../outside.npy'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='escapes'):
        ResultReader(tmp_path)


def test_exposure_only_and_invalid_time_window(tmp_path):
    w=writer(tmp_path,retention='exposure')
    append(w,0)
    assert 'values' not in w.manifest['blocks'][0]['files']
    with pytest.raises(ValueError,match='not retained'):
        ResultReader(tmp_path).read()
    with pytest.raises(ValueError,match='overlaps'):
        append(w,0)


def test_csv_weather_units_and_interval_overlap(tmp_path):
    path,spec=fixture(tmp_path)
    (tmp_path/'weather.csv').write_text('timestamp_utc,duration_hours,dni,dhi\n2025-06-21T10:07:30Z,0.25,600,100\n2025-06-21T10:22:30Z,0.25,610,110\n')
    spec['weather']={'type':'csv','path':'weather.csv','units':'W/m2'}
    path.write_text(json.dumps(spec))
    project=load_project(path)
    np.testing.assert_array_equal(project.dni,[600,610])
    spec['weather']['units']='kW/m2'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='units'):
        load_project(path)


@pytest.mark.parametrize('payload,match', [
    ('id,group,x,y,z,nx,ny,nz\na,g,0,0,1,0,0,0\n', 'normals'),
    ('id,group,x,y,z,nx,ny,nz\na,g,0,0,1,0,0,1\na,g,0,0,1,0,0,1\n', 'unique'),
    ('id,group,x,y,z,nx,ny,nz\na,g,nan,0,1,0,0,1\n', 'finite'),
])
def test_invalid_sensor_tables(tmp_path,payload,match):
    path,_=fixture(tmp_path)
    (tmp_path/'sensors.csv').write_text(payload)
    with pytest.raises(ValueError,match=match):
        load_project(path)


def test_invalid_geometry_dtype_and_deformation_count(tmp_path):
    path,spec=fixture(tmp_path)
    np.save(tmp_path/'vertices.npy',np.zeros((3,3),np.float64))
    with pytest.raises(ValueError,match='float32'):
        load_project(path)
    np.save(tmp_path/'vertices.npy',np.zeros((3,3),np.float32))
    np.save(tmp_path/'deform.npy',np.zeros((4,3),np.float32))
    spec['states'][0].pop('transforms')
    spec['states'][0]['vertices']={'one':'deform.npy'}
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='vertex count'):
        load_project(path)


def test_overlapping_and_negative_weather(tmp_path):
    path,spec=fixture(tmp_path)
    spec['states']=[]
    spec['weather']={'type':'csv','path':'weather.csv','units':'W/m2'}
    path.write_text(json.dumps(spec))
    table=tmp_path/'weather.csv'
    table.write_text('timestamp_utc,duration_hours,dni,dhi\n2025-06-21T10:00:00Z,1,600,100\n2025-06-21T10:15:00Z,1,600,100\n')
    with pytest.raises(ValueError,match='overlap'):
        load_project(path)
    table.write_text('timestamp_utc,duration_hours,dni,dhi\n2025-06-21T10:00:00Z,1,-1,100\n')
    with pytest.raises(ValueError,match='nonnegative'):
        load_project(path)


def test_store_negative_inputs_do_not_publish(tmp_path):
    w=writer(tmp_path)
    for stamps,durations,values in [
        (np.array(['NaT'],dtype='datetime64[us]'),[1],np.ones((1,3,2,1))),
        (np.array(['2025-01-01'],dtype='datetime64[us]'),[0],np.ones((1,3,2,1))),
        (np.array(['2025-01-01'],dtype='datetime64[us]'),[1],-np.ones((1,3,2,1))),
    ]:
        with pytest.raises(ValueError):
            w.append(stamps,durations,values,np.zeros_like(values),0,0)
        assert ResultReader(tmp_path).manifest['completed_steps']==0
    append(w,0)
    reader=ResultReader(tmp_path)
    for kwargs in [dict(start=2,stop=1),dict(stop=3),dict(sensor_ids=['unknown']),dict(components=['sky','sky'])]:
        with pytest.raises(ValueError):
            reader.read(**kwargs)


@pytest.mark.parametrize('value',['2025-01-01T00:00:00+02:00Z','2025-01-01','NaTZ','2025-01-01T00:00:00+00:00'])
def test_utc_rejects_ambiguous_zone_forms(value):
    with pytest.raises(ValueError,match='UTC'):
        utc_timestamp(value)


def test_failed_publication_leaves_prior_block_readable_and_resumable(tmp_path,monkeypatch):
    import raystrack.project_store as module
    w=writer(tmp_path)
    first,_=append(w,0)
    original=module._publish
    def fail(root,manifest):
        raise OSError('simulated publication interruption')
    monkeypatch.setattr(module,'_publish',fail)
    with pytest.raises(OSError,match='interruption'):
        append(w,2)
    np.testing.assert_array_equal(ResultReader(tmp_path).read()['values'],first)
    monkeypatch.setattr(module,'_publish',original)
    resumed=writer(tmp_path,resume=True)
    assert resumed.completed_steps==2
    append(resumed,2)
    assert len(ResultReader(tmp_path).read()['timestamps'])==4


def test_manifest_axis_and_chunk_metadata_rejected(tmp_path):
    w=writer(tmp_path)
    append(w,0)
    manifest=w.manifest
    manifest['axes']['sensors']=['a','a','c']
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='unique'):
        ResultReader(tmp_path)


@pytest.mark.parametrize('retention',['hourly','exposure'])
def test_resume_rejects_tampered_integrated_summary(tmp_path,retention):
    w=writer(tmp_path,retention=retention)
    append(w,0)
    manifest=w.manifest
    manifest['exposure_wh_m2'][0][0][0]+=1
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='aggregate'):
        writer(tmp_path,resume=True,retention=retention)


def test_time_metadata_cannot_hide_selected_samples(tmp_path):
    w=writer(tmp_path)
    append(w,0)
    manifest=w.manifest
    manifest['blocks'][0]['start_utc']='2025-02-01T00:00:00Z'
    manifest['blocks'][0]['end_utc']='2025-02-01T02:00:00Z'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='temporal'):
        ResultReader(tmp_path).read_times('2025-01-01T00:00:00Z','2025-01-01T02:00:00Z')
    with pytest.raises(ValueError,match='temporal'):
        writer(tmp_path,resume=True)
