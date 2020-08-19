import unittest
from lenses import mbx
import torch
import torch.nn as nn
import os
import tempfile


class TestExplorer(unittest.TestCase):
    def test_add_module(self):
        explorer = mbx.Explorer()
        self.assertEqual(len(explorer.hooks), 0)
        self.assertEqual(len(explorer.modules), 0)
        m = nn.Linear(4, 5)
        self.assertEqual(len(m._forward_hooks), 0)
        explorer.add_module("fc1", m)
        self.assertEqual(len(explorer.modules), 1)

        # adding a module to the explorer
        # should not register any hooks yet
        self.assertEqual(len(explorer.hooks), 0)
        self.assertEqual(len(m._forward_hooks), 0)

        explorer.register_hooks()
        self.assertEqual(len(explorer.hooks), 1)
        self.assertEqual(len(m._forward_hooks), 1)

        explorer.remove_hooks()
        self.assertEqual(len(m._forward_hooks), 0)
        self.assertEqual(len(explorer.hooks), 0)

    def test_record_outputs(self):
        torch.manual_seed(0)

        explorer = mbx.Explorer()

        input_features, output_features = 4, 5
        m = nn.Linear(input_features, output_features)
        module = explorer.add_module("fc1", m)

        num_samples = 50
        x = torch.randn(num_samples, input_features)
        m(x)

        # no hooks, no outputs
        self.assertEqual(len(module.outputs), 0)

        explorer.register_hooks()
        m(x)

        self.assertEqual(len(module.outputs), num_samples)
        self.assertEqual(module.outputs[0].shape, (output_features,))

    def test_compute_pcs(self):
        torch.manual_seed(0)

        explorer = mbx.Explorer()

        input_features, output_features = 12, 9
        m = nn.Linear(input_features, output_features)
        module = explorer.add_module("fc1", m)

        explorer.register_hooks()

        num_samples = 25
        x = torch.randn(num_samples, input_features)
        m(x)

        self.assertEqual(len(module.outputs), num_samples)
        self.assertEqual(len(module.pcs), 0)
        module.compute_pcs(5)
        self.assertEqual(len(module.pcs), 5)

        # cannot compute more pcs than output features
        self.assertEqual(len(module.outputs), num_samples)
        module.compute_pcs(20)
        self.assertEqual(len(module.pcs), output_features)

        # cannot compute more pcs than available samples
        num_samples = 3
        x = torch.randn(num_samples, input_features)
        module.clear()
        m(x)
        self.assertEqual(len(module.outputs), num_samples)
        with self.assertRaises(ValueError):
            module.compute_pcs(4)

    def test_compute_pc_correlations(self):
        torch.manual_seed(0)

        explorer = mbx.Explorer()

        m1 = nn.Linear(12, 5)
        module1 = explorer.add_module("fc1", m1)

        m2 = nn.Linear(5, 9)
        module2 = explorer.add_module("fc2", m2)

        explorer.register_hooks()
        self.assertEqual(len(explorer.hooks), 2)

        num_samples = 25
        x = torch.randn(num_samples, 12)
        x1 = m1(x)
        m2(x1)

        self.assertEqual(len(module1.outputs), num_samples)
        self.assertEqual(module1.outputs[0].shape, (5,))

        self.assertEqual(len(module2.outputs), num_samples)
        self.assertEqual(module2.outputs[0].shape, (9,))

        self.assertEqual(len(module1.pcs), 0)
        self.assertEqual(len(module2.pcs), 0)
        explorer.compute_pcs(5)
        self.assertEqual(len(module1.pcs), 5)
        self.assertEqual(len(module2.pcs), 5)

        self.assertEqual(len(explorer.pc_correlations), 0)
        explorer.compute_pc_correlations()
        self.assertEqual(len(explorer.pc_correlations), 25)

    def test_dict(self):
        torch.manual_seed(0)

        src_explorer = mbx.Explorer()

        output_features = 10
        model = nn.Sequential(
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, output_features)
        )

        src_explorer.add_module("fc1", model[0])
        src_explorer.add_module("fc2", model[2])
        self.assertEqual(len(src_explorer.modules), 2)

        src_explorer.register_hooks()
        num_samples = 100
        x = torch.randn(num_samples, output_features)
        model(x)
        src_explorer.remove_hooks()

        num_components = 5
        src_explorer.compute_pcs(num_components)
        self.assertEqual(len(src_explorer.modules[0].pcs), num_components)
        self.assertEqual(len(src_explorer.modules[1].pcs), num_components)

        self.assertEqual(len(src_explorer.pc_correlations), 0)
        src_explorer.compute_pc_correlations()
        self.assertEqual(len(src_explorer.pc_correlations), 25)

        data = src_explorer.to_dict()

        explorer = mbx.Explorer.from_dict(data)

        self.assertEqual(len(src_explorer.modules), len(explorer.modules))

        for src_module, module in zip(src_explorer.modules, explorer.modules):
            self.assertEqual(len(src_module.name), len(module.name))
            self.assertEqual(len(src_module.pcs), len(module.pcs))
            for src_pc, pc in zip(src_module.pcs, module.pcs):
                self.assertEqual(src_pc.value, pc.value)

        self.assertEqual(
            len(src_explorer.pc_correlations), len(explorer.pc_correlations)
        )

        for src_correlation, correlation in zip(
            src_explorer.pc_correlations, explorer.pc_correlations
        ):
            self.assertEqual(src_correlation.a.id, correlation.a.id)
            self.assertEqual(src_correlation.b.id, correlation.b.id)
            self.assertEqual(src_correlation.value, correlation.value)

    def test_save_load(self):
        torch.manual_seed(0)

        src_explorer = mbx.Explorer()

        output_features = 10
        model = nn.Sequential(
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, output_features)
        )

        src_explorer.add_module("fc1", model[0])
        src_explorer.add_module("fc2", model[2])
        self.assertEqual(len(src_explorer.modules), 2)

        src_explorer.register_hooks()
        num_samples = 100
        x = torch.randn(num_samples, output_features)
        model(x)
        src_explorer.remove_hooks()

        num_components = 5
        src_explorer.compute_pcs(num_components)
        self.assertEqual(len(src_explorer.modules[0].pcs), num_components)
        self.assertEqual(len(src_explorer.modules[1].pcs), num_components)

        src_explorer.compute_ics(num_components)
        self.assertEqual(len(src_explorer.modules[0].ics), num_components)
        self.assertEqual(len(src_explorer.modules[1].ics), num_components)

        src_explorer.compute_pc_correlations()
        src_explorer.compute_ic_correlations()

        self.assertEqual(len(src_explorer.pc_correlations), 25)
        self.assertEqual(len(src_explorer.ic_correlations), 25)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            dirname = os.path.join(tmp_dirname, "explorer")

            src_explorer.save(dirname)
            explorer = mbx.Explorer.load(dirname)

            self.assertEqual(len(src_explorer.modules), len(explorer.modules))

            for src_module, module in zip(src_explorer.modules, explorer.modules):
                self.assertEqual(len(src_module.pcs), len(module.pcs))
                for src_pc, pc in zip(src_module.pcs, module.pcs):
                    self.assertEqual(len(src_pc.samples), len(pc.samples))
                for src_ic, ic in zip(src_module.ics, module.ics):
                    self.assertEqual(len(src_ic.samples), len(ic.samples))

    def test_save_overwrite(self):
        explorer = mbx.Explorer()

        with tempfile.TemporaryDirectory() as tmp_dirname:
            dirname = os.path.join(tmp_dirname, "workspace")
            explorer.save(dirname, overwrite=False)

            with self.assertRaises(RuntimeError):
                explorer.save(dirname, overwrite=False)

            explorer.save(dirname, overwrite=True)

    def test_ica(self):
        torch.manual_seed(0)

        explorer = mbx.Explorer()
        num_samples = 100
        input_size = 30
        fc = nn.Linear(input_size, 20)
        module = explorer.add_module("fc", fc)
        x = torch.randn(num_samples, input_size)
        explorer.register_hooks()
        fc(x)

        num_components = 5
        self.assertEqual(len(module.ics), 0)
        explorer.compute_ics(num_components)
        explorer.remove_hooks()

        self.assertEqual(len(module.ics), num_components)


if __name__ == "__main__":
    unittest.main()
