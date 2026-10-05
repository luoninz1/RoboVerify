"""The three historical top-down datasets, retained unchanged as a regression corpus."""


class Box:
    def __init__(self, id):
        self.id = id

    def set_attribute(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def run_stack_hardcoded_dataset():
    all_inputs = []
    # scene 1
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.0)

    b2 = Box("2")
    b2.set_attribute(1.0, 0.0, 0.0)

    b3 = Box("3")
    b3.set_attribute(2.0, 0.0, 0.0)

    b4 = Box("4")
    b4.set_attribute(3.0, 0.0, 0.0)

    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 2
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.0)

    b2 = Box("2")
    b2.set_attribute(1.0, 0.0, 0.0)

    b3 = Box("3")
    b3.set_attribute(3.0, 0.0, 0.05)

    b4 = Box("4")
    b4.set_attribute(3.0, 0.0, 0.0)

    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b3},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b3},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b3},
            "result": False,
        }
    )

    # scene 3
    b5 = Box("5")
    b5.set_attribute(0.0, 0.0, 0.1)

    b0 = Box("0")
    b0.set_attribute(0.0, 0.0, 0.05)

    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.0)

    b2 = Box("2")
    b2.set_attribute(3.0, 0.0, 0.1)

    b3 = Box("3")
    b3.set_attribute(3.0, 0.0, 0.05)

    b4 = Box("4")
    b4.set_attribute(3.0, 0.0, 0.0)

    all_inputs.append(
        {
            "target": b5,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b0,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b1,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    return all_inputs


def run_unstack_hardcoded_dataset():
    all_inputs = []
    # scene 1
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.2)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 2
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 3
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(4.0, 0.0, 0.05)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 4
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(4.0, 0.0, 0.05)

    b3 = Box("3")
    b3.set_attribute(6.0, 0.0, 0.05)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    return all_inputs


def run_reverse_hardcoded_dataset():
    all_inputs = []
    # scene 1
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.2)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 2
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 3
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(2.0, 0.0, 0.1)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 4
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(2.0, 0.0, 0.1)

    b3 = Box("3")
    b3.set_attribute(2.0, 0.0, 0.15)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 5
    # b1 = Box("1")
    # b1.set_attribute(2.0, 0.0, 0.05)

    # b2 = Box("2")
    # b2.set_attribute(2.0, 0.0, 0.1)

    # b3 = Box("3")
    # b3.set_attribute(2.0, 0.0, 0.15)

    # b4 = Box("4")
    # b4.set_attribute(2.0, 0.0, 0.2)

    # all_inputs.append(
    #     {
    #         "target": b1,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    # all_inputs.append(
    #     {
    #         "target": b2,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    # all_inputs.append(
    #     {
    #         "target": b3,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    # all_inputs.append(
    #     {
    #         "target": b4,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    return all_inputs
