import { calcHSLFromScore } from "../utils/color";
import React from "react";
import styles from "../App.module.css";
import Tooltip from "./Tooltip";
import Plot from "./Plot";
import { FeatureOutput } from "../models/visualizationOutput";

interface FeatureProps<T> {
  data: T;
}

type ImageFeatureProps = FeatureProps<{
  base: string;
  modified: string;
  name: string;
}>;

function ImageFeature(props: ImageFeatureProps) {
  return (
    <>
      <div className={styles["panel__column__title"]}>
        {props.data.name} (Image)
      </div>
      <div className={styles["panel__column__body"]}>
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
      <div className={styles["panel__column__title"]}>
        {props.data.name} (Text)
      </div>
      <div className={styles["panel__column__body"]}>{color_words}</div>
    </>
  );
}

type GeneralFeatureProps = FeatureProps<{
  base: number[];
  modified: number[];
  name: string;
}>;

function GeneralFeature(props: GeneralFeatureProps) {
  return (
    <Plot
      data={[
        {
          x: props.data.base,
          y: props.data.modified,
          type: "bar",
          marker: {
            color: props.data.modified.map(
              (v) => (v < 0 ? "#d45c43" : "#80aaff") // red if negative, else blue
            ),
          },
        },
      ]}
      config={{
        displayModeBar: false,
      }}
      layout={{
        height: 300,
        margin: {
          t: 20,
          pad: 0,
        },
        yaxis: {
          fixedrange: true,
          showgrid: false,
        },
        xaxis: {
          fixedrange: false,
        },
      }}
    />
  );
}

function Feature(props: {data: FeatureOutput}) {
  const data = props.data;
  switch (data.type) {
    case "image":
      return <ImageFeature data={data} />;
    case "text":
      return <TextFeature data={data} />;
    case "general":
      return <GeneralFeature data={data} />;
    case "empty":
      return <></>;
    default:
      throw new Error("Unsupported feature visualization type: " + data.type);
  }
}

export default Feature;
