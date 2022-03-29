import json
from os import path

import autocti as ac

json_file = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "file.json"
)


def test__trap_instant_capture__dictable():

    trap = ac.TrapInstantCapture(density=1.0, release_timescale=2.0)

    with open(json_file, "w+") as f:
        json.dump(trap.dict(), f, indent=4)

    with open(json_file, "r+") as f:
        trap_load_dict = json.load(f)

    trap_load = ac.TrapInstantCapture.from_dict(trap_load_dict)

    assert trap_load.density == 1.0
    assert trap_load.release_timescale == 2.0
