import unittest
from lenses import mbx
import torch
import torch.nn as nn
import os
import tempfile


class TestModule(unittest.TestCase):
    def test_save_load_outputs(self):
        torch.manual_seed(0)

        num_samples = 100
        input_size = 5
        output_size = 3

        fc = nn.Linear(input_size, output_size)

        # the user would normally create modules using explorer.add_module,
        # but setting explorer to None here is useful
        # to test module functionality only
        explorer = None
        id = 0
        src_module = mbx.Module(explorer, id, "fc1", fc)

        src_module.outputs = [torch.randn(output_size) for _ in range(num_samples)]

        # save outputs = True
        with tempfile.TemporaryDirectory() as tmp_dirname:
            module_dirname = os.path.join(tmp_dirname, "module")
            src_module.save(module_dirname, save_outputs=True)

            module = mbx.Module.load(module_dirname, explorer, id, load_outputs=True)
            self.assertEqual(src_module.name, module.name)
            self.assertEqual(len(module.outputs), num_samples)

            module = mbx.Module.load(module_dirname, explorer, id, load_outputs=False)
            self.assertEqual(src_module.name, module.name)
            self.assertEqual(len(module.outputs), 0)

        # save outputs = False
        with tempfile.TemporaryDirectory() as tmp_dirname:
            module_dirname = os.path.join(tmp_dirname, "module")
            src_module.save(module_dirname, save_outputs=False)

            module = mbx.Module.load(module_dirname, explorer, id, load_outputs=False)
            self.assertEqual(src_module.name, module.name)
            self.assertEqual(len(module.outputs), 0)

            with self.assertRaises(RuntimeError):
                module = mbx.Module.load(module_dirname, explorer, id, load_outputs=True)

        # overwrite test
        with tempfile.TemporaryDirectory() as tmp_dirname:
            module_dirname = os.path.join(tmp_dirname, "module")

            src_module.save(module_dirname, overwrite=False, save_outputs=True)
            self.assertTrue(os.path.exists(module_dirname))

            with self.assertRaises(RuntimeError):
                src_module.save(module_dirname, overwrite=False, save_outputs=True)

            src_module.save(module_dirname, overwrite=True, save_outputs=True)
            self.assertTrue(os.path.exists(module_dirname))


if __name__ == "__main__":
    unittest.main()
