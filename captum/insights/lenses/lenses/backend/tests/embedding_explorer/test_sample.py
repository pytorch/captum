import unittest
from lenses import mbx


class TestModule(unittest.TestCase):
    def test_generic_sample_dict(self):
        class Sample(mbx.sample.GenericSample):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.attr_request_handlers.enable_image_from_file(
                    "picture", "/data/pictures/full_size", "picture.png"
                )

            def to_payload(self):
                return {"picture": None, "name": self.name}

        name = "my name"
        sample = Sample(name)
        sample_dict = sample.to_dict()
        self.assertEqual(sample_dict["payload"]["name"], name)
        self.assertEqual(
            sample_dict["payload"]["picture"],
            {"type": mbx.sample.GenericSampleAttrType.IMAGE.value},
        )


if __name__ == "__main__":
    unittest.main()
