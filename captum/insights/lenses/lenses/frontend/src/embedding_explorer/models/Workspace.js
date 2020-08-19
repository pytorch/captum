import Explorer from "./Explorer";
import Correlations from "./Correlations";
import { ComponentType } from "./Component";

function componentFullIdToNumList(fullId) {
  return [fullId.module.explorer.id, fullId.module.id, fullId.id];
}

function moduleFullIdToNumList(fullId) {
  return [fullId.explorer.id, fullId.id];
}

class Workspace {
  constructor(id) {
    this.id = id;
    this.explorers = [];
    this.pcCorrelations = new Correlations(componentFullIdToNumList);
    this.icCorrelations = new Correlations(componentFullIdToNumList);
    this.moduleCorrelations = new Correlations(moduleFullIdToNumList);
  }

  getComponentCorrelations(type) {
    if (type === ComponentType.PCA) {
      return this.pcCorrelations;
    } else if (type === ComponentType.ICA) {
      return this.icCorrelations;
    } else {
      throw new Error(`invalid component type ${type}`);
    }
  }

  static fromData(data) {
    const workspace = new Workspace(data.id);
    data.explorers.forEach((explorerData) => {
      const explorer = Explorer.fromData(explorerData, workspace);
      workspace.explorers.push(explorer);
    });
    if (data.module_pc_correlations != null) {
      workspace.pcCorrelations = Correlations.fromData(
        data.module_pc_correlations,
        componentFullIdToNumList,
      );
    }
    if (data.module_ic_correlations != null) {
      workspace.icCorrelations = Correlations.fromData(
        data.module_ic_correlations,
        componentFullIdToNumList,
      );
    }
    if (data.module_correlations != null) {
      workspace.moduleCorrelations = Correlations.fromData(
        data.module_correlations,
        moduleFullIdToNumList,
      );
    }
    return workspace;
  }
}

export default Workspace;
