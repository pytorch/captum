import * as chartjs from "chart.js";

// Because there's no data point type exported by the
// main type declaration for chart.js, we have our own.

export interface DataPoint {
    chart?: object;
    dataIndex?: number;
    dataset?: chartjs.ChartDataSets;
    datasetIndex?: number;
}
