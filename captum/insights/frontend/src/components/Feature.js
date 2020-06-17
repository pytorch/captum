import { calcHSLFromScore } from "../utils/color";
import React from "react";
import styles from "../App.module.css";
import Tooltip from "./Tooltip";
import Plot from "./Plot";

function ImageFeature(props) {
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

function TextFeature(props) {
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
          <Tooltip label={props.data.modified[i].toFixed(3)} />
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

function GeneralFeature(props) {
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
          title: false,
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

function Feature(props) {
  const data = props.data;
  switch (data.type) {
    case "image":
      return <ImageFeature data={data} />;
    case "text":
      return <TextFeature data={data} />;
    case "general":
      return <GeneralFeature data={data} />;
    default:
      throw new Error("Unsupported feature visualization type: " + data.type);
  }
}

export default Feature;
