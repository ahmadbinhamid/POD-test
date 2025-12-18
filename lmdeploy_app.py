import PIL
from lmdeploy import api
from lmdeploy import TurbomindEngineConfig


backend_config = TurbomindEngineConfig(
    tp=1,
    max_batch_size=4,
    # This parameter is also important for managing concurrent users.
    # It controls the percentage of GPU memory allocated for the K/V cache.
    # A higher value allows more concurrent conversations.

)

client = api.serve(
    model_name="InternVL2-8B",
    model_path="OpenGVLab/InternVL2-8B",
    server_name="0.0.0.0",
    server_port=23333,
    backend_config=backend_config
)


while True:
    client
 