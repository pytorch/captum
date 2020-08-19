import Module from "./Module";

export class SampleBin {
  constructor(component, id, begin, end, sortedSampleIds) {
    this.component = component;
    this.id = id;
    this.begin = begin;
    this.end = end;
    this.sortedSampleIds = sortedSampleIds;
  }

  prev() {
    if (this.id === 0) return null;
    else return this.component.sampleBins[this.id - 1];
  }

  next() {
    if (this.id < this.component.sampleBins.length - 1)
      return this.component.sampleBins[this.id + 1];
    else return null;
  }

  getSampleWindow(i, size, anchor, useOtherBins) {
    if (useOtherBins == null) useOtherBins = true;
    if (size < 0) throw new Error("size must be non-negative");
    if (
      (i >= this.sortedSampleIds.length || i < 0) &&
      this.sortedSampleIds.length !== 0
    )
      throw new Error("index is out of valid range");

    let sampleIds = [];

    if (anchor === "center") {
      const iLeft = i - Math.floor(size * 0.5);
      const iRight = iLeft + size;

      if (iLeft >= 0 && iRight <= this.sortedSampleIds.length) {
        sampleIds = this.sortedSampleIds.slice(iLeft, iRight);
      } else {
        let sampleIdsLeft = [];
        let sampleIdsRight = [];
        if (iLeft < 0 && useOtherBins) {
          const prevBin = this.prev();
          if (prevBin != null)
            sampleIdsLeft = prevBin.getSampleWindow(
              0,
              -iLeft,
              "end",
              useOtherBins,
            );
        }
        if (iRight > this.sortedSampleIds.length && useOtherBins) {
          const nextBin = this.next();
          if (nextBin != null)
            sampleIdsRight = nextBin.getSampleWindow(
              0,
              iRight - this.sortedSampleIds.length,
              "begin",
              useOtherBins,
            );
        }
        const sampleIdsMid = this.sortedSampleIds.slice(
          Math.max(0, iLeft),
          Math.min(this.sortedSampleIds.length, iRight),
        );
        sampleIds = sampleIdsLeft.concat(sampleIdsMid).concat(sampleIdsRight);
      }
    } else if (anchor === "end") {
      if (size <= this.sortedSampleIds.length) {
        sampleIds = this.sortedSampleIds.slice(-size);
      } else {
        sampleIds = this.sortedSampleIds.slice();
        const prevBin = this.prev();
        if (prevBin != null && useOtherBins) {
          sampleIds = prevBin
            .getSampleWindow(
              0,
              size - this.sortedSampleIds.length,
              "end",
              useOtherBins,
            )
            .concat(sampleIds);
        }
      }
    } else if (anchor === "begin") {
      if (size <= this.sortedSampleIds.length) {
        sampleIds = sampleIds.concat(this.sortedSampleIds.slice(0, size));
      } else {
        sampleIds = this.sortedSampleIds.slice();
        const nextBin = this.next();
        if (nextBin != null && useOtherBins) {
          sampleIds = sampleIds.concat(
            nextBin.getSampleWindow(
              0,
              size - this.sortedSampleIds.length,
              "begin",
              useOtherBins,
            ),
          );
        }
      }
    } else {
      throw new Error(`unknown anchor ${anchor}`);
    }

    return sampleIds;
  }

  static fromData(data, component, id) {
    const sampleBin = new SampleBin(
      component,
      id,
      data.begin,
      data.end,
      data.sorted_sample_ids,
    );
    return sampleBin;
  }
}

class Component {
  constructor(module, id, value, type) {
    this.module = module;
    this.id = id;
    this.value = value;
    this.type = type;
    this.sampleBins = [];
  }

  fullId() {
    return { id: this.id, module: this.module.fullId() };
  }

  static equalFullIds(a, b) {
    return a.id === b.id && Module.equalFullIds(a.module, b.module);
  }

  equals(b) {
    return (
      b instanceof Component &&
      Component.equalFullIds(this.fullId(), b.fullId()) &&
      b.type === this.type
    );
  }

  getCorrelation(b) {
    if (!(b instanceof Component)) return null;

    // TODO correlation between different component types might be useful at some point,
    // but at the moment we do not have a use case for it
    if (this.type !== b.type) return null;

    if (this.module.explorer.equals(b.module.explorer)) {
      const correlations = this.module.explorer.getComponentCorrelations(
        this.type,
      );
      return correlations.get(this, b);
    } else {
      const workspace = this.module.explorer.workspace;
      const correlations = workspace.explorer.getComponentCorrelations(
        this.type,
      );
      return correlations.get(this, b);
    }
  }

  static fromData(data, module, type) {
    const component = new Component(module, data.id, data.value, type);
    component.sampleBins = data.sample_bins.map((sampleBinData, i) =>
      SampleBin.fromData(sampleBinData, component, i),
    );
    return component;
  }
}

export class ComponentType {
  constructor(id, modeName, componentPrefix) {
    this.id = id;
    this.modeName = modeName;
    this.componentPrefix = componentPrefix;
  }

  toggle() {
    if (this === ComponentType.PCA) return ComponentType.ICA;
    else return ComponentType.PCA;
  }
}

ComponentType.PCA = new ComponentType(0, "PCA", "PC");
ComponentType.ICA = new ComponentType(1, "ICA", "IC");

export default Component;
