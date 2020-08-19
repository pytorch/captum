import Component, { ComponentType, SampleBin } from "./Component";

it("implements sample bin logic correctly", () => {
  const module = null;
  const component = new Component(module, 0, 0.9, ComponentType.PCA);

  expect(component.value).toBe(0.9);

  const bin0 = new SampleBin(component, 0, 0, 0.5, [4, 10, 8]);
  component.sampleBins.push(bin0);
  const bin1 = new SampleBin(component, 1, 0.5, 1.0, [11, 20, 9, 18]);
  component.sampleBins.push(bin1);
  const bin2 = new SampleBin(component, 2, 1.0, 1.5, [1]);
  component.sampleBins.push(bin2);

  expect(bin0.next()).toBe(bin1);
  expect(bin0.prev()).toBe(null);

  expect(bin1.next()).toBe(bin2);
  expect(bin1.prev()).toBe(bin0);

  expect(bin2.next()).toBe(null);
  expect(bin2.prev()).toBe(bin1);

  // getSampleWindow tests

  //  bin0  |    bin1     | bin2
  // 4 10 8 | 11 20 9 18  |  1

  let sampleIds;

  sampleIds = bin0.getSampleWindow(0, 4, "begin", false);
  expect(sampleIds).toEqual([4, 10, 8]);

  sampleIds = bin0.getSampleWindow(0, 4, "begin", true);
  expect(sampleIds).toEqual([4, 10, 8, 11]);

  sampleIds = bin2.getSampleWindow(0, 1, "end", true);
  expect(sampleIds).toEqual([1]);

  sampleIds = bin2.getSampleWindow(0, 3, "end", true);
  expect(sampleIds).toEqual([9, 18, 1]);

  sampleIds = bin1.getSampleWindow(1, 3, "center", true);
  expect(sampleIds).toEqual([11, 20, 9]);

  sampleIds = bin1.getSampleWindow(0, 3, "center", true);
  expect(sampleIds).toEqual([8, 11, 20]);

  sampleIds = bin1.getSampleWindow(0, 3, "center", false);
  expect(sampleIds).toEqual([11, 20]);

  sampleIds = bin1.getSampleWindow(3, 3, "center", true);
  expect(sampleIds).toEqual([9, 18, 1]);

  sampleIds = bin1.getSampleWindow(3, 5, "center", true);
  expect(sampleIds).toEqual([20, 9, 18, 1]);

  expect(() => {
    sampleIds = bin1.getSampleWindow(0, -5, "center", true);
  }).toThrow();

  expect(() => {
    sampleIds = bin1.getSampleWindow(1000, 1, "center", true);
  }).toThrow();
});

it("implements sample bin logic correctly when empty bins are present", () => {
  const module = null;
  const component = new Component(module, 0, 0.9, ComponentType.PCA);

  expect(component.value).toBe(0.9);

  const bin0 = new SampleBin(component, 0, 0, 0.5, [4, 10, 8]);
  component.sampleBins.push(bin0);
  const bin1 = new SampleBin(component, 1, 0.5, 1.0, [11, 20, 9, 18]);
  component.sampleBins.push(bin1);
  const bin2 = new SampleBin(component, 2, 1.0, 1.5, []);
  component.sampleBins.push(bin2);
  const bin3 = new SampleBin(component, 3, 1.0, 1.5, []);
  component.sampleBins.push(bin3);
  const bin4 = new SampleBin(component, 4, 1.0, 1.5, [1]);
  component.sampleBins.push(bin4);

  // getSampleWindow tests

  //  bin0  |    bin1     | bin2 | bin3 | bin4
  // 4 10 8 | 11 20 9 18  |      |      |  1

  let sampleIds;
  sampleIds = bin4.getSampleWindow(0, 3, "center", true);
  expect(sampleIds).toEqual([18, 1]);

  sampleIds = bin4.getSampleWindow(0, 11, "center", true);
  expect(sampleIds).toEqual([8, 11, 20, 9, 18, 1]);

  sampleIds = bin4.getSampleWindow(0, 11, "center", false);
  expect(sampleIds).toEqual([1]);
});
