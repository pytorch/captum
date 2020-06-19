interface OutputScore {
    label: string;
    index: number;
    score: number;
}

interface FeatureOutput {
    name: string;
    type: string;
    base: string;
    contribution: number;
    modified: string;
}

export interface VisualizationOutput {
    feature_outputs: FeatureOutput[];
    actual: OutputScore;
    predicted: OutputScore[];
    active_index: number;
}
