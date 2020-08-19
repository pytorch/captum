import { hashNumListPair } from "../utils/hash";

class Correlations {
  constructor(componentFullIdToNumList) {
    this.keyToCorrelation = {};
    this.componentFullIdToNumList = componentFullIdToNumList;
  }

  get(component1, component2) {
    const [a, b] = [component1, component2].map((c) =>
      this.componentFullIdToNumList(c.fullId()),
    );
    const key = hashNumListPair(a, b);
    return this.keyToCorrelation[key];
  }

  static fromData(data, componentFullIdToNumList) {
    const correlations = new Correlations(componentFullIdToNumList);
    data.forEach((correlation) => {
      const [a, b] = [correlation.a, correlation.b].map(
        correlations.componentFullIdToNumList,
      );
      const key = hashNumListPair(a, b);
      correlations.keyToCorrelation[key] = correlation.value;
    });
    return correlations;
  }
}

export default Correlations;
