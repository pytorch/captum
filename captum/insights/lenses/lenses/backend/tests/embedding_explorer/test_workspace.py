import unittest
from lenses import mbx
import torch
import torch.nn as nn
import os
import tempfile


class TestWorkspace(unittest.TestCase):
    def test_add_explorer(self):
        workspace = mbx.Workspace()
        self.assertEqual(len(workspace.explorers), 0)
        explorer1 = workspace.add_explorer()
        self.assertIsInstance(explorer1, mbx.Explorer)
        self.assertEqual(len(workspace.explorers), 1)
        explorer2 = workspace.add_explorer()
        self.assertNotEqual(explorer1.id, explorer2.id)
        self.assertEqual(len(workspace.explorers), 2)

    def test_save_load(self):
        src_workspace = mbx.Workspace()
        explorer = src_workspace.add_explorer()

        explorer.add_module("fc", nn.Linear(5, 10))
        self.assertEqual(len(src_workspace.explorers), 1)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            dirname = os.path.join(tmp_dirname, "explorer")

            src_workspace.save(dirname)
            workspace = mbx.Workspace.load(dirname)

            self.assertEqual(len(src_workspace.explorers), len(workspace.explorers))

            for src_explorer, explorer in zip(
                src_workspace.explorers, workspace.explorers
            ):
                self.assertEqual(len(src_explorer.modules), len(explorer.modules))
                for src_module, module in zip(src_explorer.modules, explorer.modules):
                    self.assertEqual(src_module.name, module.name)

    def test_save_overwrite(self):
        workspace = mbx.Workspace()

        with tempfile.TemporaryDirectory() as tmp_dirname:
            dirname = os.path.join(tmp_dirname, "workspace")
            workspace.save(dirname, overwrite=False)

            with self.assertRaises(RuntimeError):
                workspace.save(dirname, overwrite=False)

            workspace.save(dirname, overwrite=True)

    def test_module_correlations(self):
        # this is a basic module correlation example where the models are exactly the same
        # and we expect a correlation of 1.0 for the corresponding modules

        torch.manual_seed(0)

        workspace = mbx.Workspace()

        input_size = 4
        output_size = 5
        num_components = output_size

        m = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )
        x = torch.randn(100, input_size, dtype=torch.float32)

        def forward():
            m(x)

        explorer = workspace.add_explorer()
        explorer.add_module("fc1", m[0])
        explorer.add_module("fc2", m[2])
        explorer.run(forward, num_components)

        explorer = workspace.add_explorer()
        explorer.add_module("fc1", m[0])
        explorer.add_module("fc2", m[2])
        explorer.run(forward, num_components)

        self.assertEqual(len(workspace.module_correlations), 0)
        workspace.compute_module_correlations(
            num_components, max_iter=100, tol=1e-3, verbose=False
        )
        self.assertEqual(len(workspace.module_correlations), 4)

        for c in workspace.module_correlations:
            if c.a.id == c.b.id:
                self.assertAlmostEqual(c.value, 1.0)
            else:
                self.assertNotAlmostEqual(c.value, 1.0)


if __name__ == "__main__":
    unittest.main()
