import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,

)

# from manipulation.utils import RenderDiagram

# Start the visualizer.
meshcat = StartMeshcat()

plant = MultibodyPlant(time_step=1e-4)
Parser(plant).AddModelsFromUrl(
    "package://drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"
)
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
plant.Finalize()