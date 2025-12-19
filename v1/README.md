# V22

- **tasks**
  - Pick up the red cuboid and place it on the cardboard box
  - Pick up the green cuboid and place it on the cardboard box
  - Pick up the blue cuboid and place it on the cardboard box

- **scene:** Kitchen. Square aspect ratio of cameras.

- **recording_step_gap:** 1

- **initial_eef_position:** Random position within high cam view.

- **high_cam_position:** Looking at the robot from the front.

- **target_position:** Cuboid position is randomly selected from 3 fixed positions, and then only the inter-item gap is randomized. The cardboard box is placed behind the target cuboid, with some randomness.

- **other_info:** Item positions are shuffled after every episode. The robot moves along straight line paths. Each episode can be split into subparts. Higher number of frames. Higher joint stiffness.
