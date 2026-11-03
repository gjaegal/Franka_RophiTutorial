# FR3 model source

Vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/b0655057a9df2abe39ea7fc989914ba6f248301c/franka_fr3), commit `b0655057a9df2abe39ea7fc989914ba6f248301c`.

`franka_fr3/fr3.xml`, its meshes, README and Apache-2.0 license are copied unchanged. `common.py` adds the cell, lighting, target and task point in memory. No Robotiq geometry, adapter or payload has been added. `index_tip` names a virtual tool point 0.107 m along the last link's local z axis, coinciding with the source model's `attachment_site`; it is not an actual index finger.
