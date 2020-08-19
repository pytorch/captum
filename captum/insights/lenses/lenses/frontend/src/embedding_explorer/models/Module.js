import Component, { ComponentType } from "./Component";
import Explorer from "./Explorer";

class Module {
  constructor(explorer, id, name) {
    this.explorer = explorer;
    this.name = name;
    this.id = id;
    this.pcs = [];
    this.ics = [];
  }

  getComponents(type) {
    if (type === ComponentType.PCA) {
      return this.pcs;
    } else if (type === ComponentType.ICA) {
      return this.ics;
    } else {
      throw new Error(`invalid component type ${type}`);
    }
  }

  fullId() {
    return { id: this.id, explorer: this.explorer.fullId() };
  }

  static equalFullIds(a, b) {
    return a.id === b.id && Explorer.equalFullIds(a.explorer, b.explorer);
  }

  static fromData(data, explorer) {
    let module = new Module(explorer, data.id, data.name);
    if (data.pcs != null) {
      module.pcs = data.pcs.map((pcData) => {
        return Component.fromData(pcData, module, ComponentType.PCA);
      });
    }
    if (data.ics != null) {
      module.ics = data.ics.map((icData) => {
        return Component.fromData(icData, module, ComponentType.ICA);
      });
    }
    return module;
  }
}

export default Module;
