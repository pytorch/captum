interface OutputScore {
  label: string;
  index: number;
  score: number;
}

export enum FeatureType {
  TEXT = "text",
  IMAGE = "image",
  GENERAL = "general",
  EMPTY = "empty",
}

type GenericFeatureOutput<F extends FeatureType, T> = {
  type: F;
  name: string;
  contribution: number;
} & T;

export type FeatureOutput =
  | GenericFeatureOutput<
      FeatureType.TEXT,
      { base: number[]; modified: number[] }
    >
  | GenericFeatureOutput<FeatureType.IMAGE, { base: string; modified: string }>
  | GenericFeatureOutput<
      FeatureType.GENERAL,
      { base: number[]; modified: number[] }
    >
  | GenericFeatureOutput<FeatureType.EMPTY, {}>;

export interface VisualizationOutput {
  model_index: number;
  feature_outputs: FeatureOutput[];
  actual: OutputScore;
  predicted: OutputScore[];
  active_index: number;
}

//When multiple models are compared, visualizations are grouped together
export type VisualizationGroup = VisualizationOutput[];
