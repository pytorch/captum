import { calcHSLFromScore } from "../utils/color";
import { DataPoint } from "../utils/dataPoint";
import React from "react";
import styles from "../App.module.css";
import Tooltip from "./Tooltip";
import { Bar } from "react-chartjs-2";
import { FeatureOutput } from "../models/visualizationOutput";

interface FeatureProps<T> {
  data: T;
  hideHeaders?: boolean;
}

type ImageFeatureProps = FeatureProps<{
  base: string;
  modified: string;
  name: string;
}>;

function ImageFeature(props: ImageFeatureProps) {
  return (
    <>
      {props.hideHeaders && (
        <div className={styles["panel__column__title"]}>
          {props.data.name} (Image)
        </div>
      )}
      <div className={styles["panel__column__body"]}>
        <div className={styles["model-number-spacer"]} />
        <div className={styles.gallery}>
          <div className={styles["gallery__item"]}>
            <div className={styles["gallery__item__image"]}>
              <img
                src={"data:image/png;base64," + props.data.base}
                alt="original"
              />
            </div>
            <div className={styles["gallery__item__description"]}>Original</div>
          </div>
          <div className={styles["gallery__item"]}>
            <div className={styles["gallery__item__image"]}>
              <img
                src={"data:image/png;base64," + props.data.modified}
                alt="attribution"
              />
            </div>
            <div className={styles["gallery__item__description"]}>
              Attribution Magnitude
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

type TextFeatureProps = FeatureProps<{
  base: number[];
  name: string;
  modified: number[];
}>;

function TextFeature(props: TextFeatureProps) {
  const color_words = props.data.base.map((w, i) => {
    return (
      <>
        <span
          style={{
            backgroundColor: calcHSLFromScore(props.data.modified[i], false),
          }}
          className={styles["text-feature-word"]}
        >
          {w}
          <Tooltip label={props.data.modified[i]?.toFixed(3)} />
        </span>{" "}
      </>
    );
  });
  return (
    <>
      {props.hideHeaders && (
        <div className={styles["panel__column__title"]}>
          {props.data.name} (Text)
        </div>
      )}
      <div className={styles["panel__column__body"]}>
        <div className={styles["model-number-spacer"]} />
        {color_words}
      </div>
    </>
  );
}

type GeneralFeatureProps = FeatureProps<{
  base: number[];
  modified: number[];
  name: string;
}>;

function GeneralFeature(props: GeneralFeatureProps) {
  const data = {
    labels: props.data.base,
    datasets: [
      {
        barPercentage: 0.5,
        data: props.data.modified,
        backgroundColor: (dataPoint: DataPoint) => {
          if (!dataPoint.dataset || !dataPoint.dataset.data || dataPoint.datasetIndex === undefined) {
            return "#d45c43"; // Default to red
          }
          const yValue = dataPoint.dataset.data[dataPoint.dataIndex as number] || 0;
          return yValue < 0 ? "#d45c43" : "#80aaff"; // Red if negative, else blue
        },
      },
    ],
  };

  return (
    <Bar
      data={data}
      width={300}
      height={50}
      legend={{ display: false }}
      options={{
        maintainAspectRatio: false,
        scales: {
          xAxes: [
            {
              gridLines: {
                display: false,
              },
            },
          ],
          yAxes: [
            {
              gridLines: {
                lineWidth: 0,
                zeroLineWidth: 1,
              },
            },
          ],
        },
      }}
    />
  );
}

function Feature(props: { data: FeatureOutput; hideHeaders: boolean }) {
  const data = props.data;
  switch (data.type) {
    case "image":
      return <ImageFeature data={data} hideHeaders={props.hideHeaders} />;
    case "text":
      return <TextFeature data={data} hideHeaders={props.hideHeaders} />;
    case "general":
      return <GeneralFeature data={data} />;
    case "empty":
      return <></>;
    default:
      throw new Error("Unsupported feature visualization type: " + data.type);
  }
}

export default Feature;
