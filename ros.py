import numpy as np

from config import (
    KEYPOINT_LAYOUT,
    KEYPOINT_TOPIC,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
)


class RosOutput:
    def __init__(self):
        import rclpy
        from std_msgs.msg import Float32MultiArray

        rclpy.init(args=None)
        self.rclpy, self.message = rclpy, Float32MultiArray
        self.node = rclpy.create_node("multiview_hand_capture")
        self.point_publisher = self.node.create_publisher(Float32MultiArray, KEYPOINT_TOPIC, 1)
        self.robot_publisher = self.node.create_publisher(Float32MultiArray, ROBOT_TOPIC, 1)
        print(f"ROS 2: publishing {KEYPOINT_TOPIC} and {ROBOT_TOPIC}")

    def publish(self, publisher, values, label, shape):
        from std_msgs.msg import MultiArrayDimension, MultiArrayLayout

        dimensions, stride = [], int(np.prod(shape))
        labels = ("keypoint", "xyz") if len(shape) == 2 else ("joint",)
        for name, size in zip(labels, shape):
            dimensions.append(
                MultiArrayDimension(
                    label=label if not dimensions else name,
                    size=size,
                    stride=stride,
                )
            )
            stride //= size
        message = self.message()
        message.layout = MultiArrayLayout(dim=dimensions, data_offset=0)
        message.data = np.asarray(values, np.float32).ravel().tolist()
        publisher.publish(message)

    def points(self, points, handedness):
        self.publish(
            self.point_publisher,
            points,
            f"{KEYPOINT_LAYOUT}:hand={handedness}",
            (21, 3),
        )

    def joints(self, degrees):
        self.publish(self.robot_publisher, degrees, ROBOT_LAYOUT, (21,))

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()
