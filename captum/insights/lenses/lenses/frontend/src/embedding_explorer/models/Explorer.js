import Module from "./Module";
import Correlations from "./Correlations";
import { ComponentType } from "./Component";

function componentFullIdToNumList(fullId) {
  return [fullId.module.id, fullId.id];
}

class Explorer {
  constructor(workspace, id, name) {
    this.workspace = workspace;
    this.id = id;
    this.name = name;
    this.modules = [];
    this.pcCorrelations = new Correlations(componentFullIdToNumList);
    this.icCorrelations = new Correlations(componentFullIdToNumList);
  }

  getComponentCorrelations(type) {
    if (type === ComponentType.PCA) {
      return this.pcCorrelations;
    } else if (type === ComponentType.ICA) {
      return this.icCorrelations;
    } else {
      throw new Error(`invalid component mode ${type}`);
    }
  }

  fullId() {
    return { id: this.id };
  }

  static equalFullIds(a, b) {
    return a.id === b.id;
  }

  equals(b) {
    return (
      b instanceof Explorer && Explorer.equalFullIds(this.fullId(), b.fullId())
    );
  }

  static fromData(data, workspace) {
    const explorer = new Explorer(workspace, data.id, data.name);
    if (data.modules != null) {
      explorer.modules = data.modules.map((moduleData) => {
        return Module.fromData(moduleData, explorer);
      });
    }
    if (data.pc_correlations != null) {
      explorer.pcCorrelations = Correlations.fromData(
        data.pc_correlations,
        componentFullIdToNumList,
      );
    }
    if (data.ic_correlations != null) {
      explorer.icCorrelations = Correlations.fromData(
        data.ic_correlations,
        componentFullIdToNumList,
      );
    }
    return explorer;
  }
}

export default Explorer;
