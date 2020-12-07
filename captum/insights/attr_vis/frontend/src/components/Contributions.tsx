import React from "react";
import styles from "../App.module.css";
import { calcHSLFromScore } from "../utils/color";
import { FeatureOutput } from "../models/visualizationOutput";

interface ContributionsProps {
  feature_outputs: FeatureOutput[];
}

function Contributions(props: ContributionsProps) {
  return (
    <>
      {props.feature_outputs.map((f) => {
        // pad bar height so features with 0 contribution can still be seen
        // in graph
        const contribution = f.contribution * 100;
        const bar_height = contribution > 10 ? contribution : contribution + 10;
        return (
          <div className={styles["bar-chart__group"]}>
            <div
              className={styles["bar-chart__group__bar"]}
              style={{
                height: bar_height + "px",
                backgroundColor: calcHSLFromScore(contribution),
              }}
            />
            <div className={styles["bar-chart__group__title"]}>{f.name}</div>
          </div>
        );
      })}
    </>
  );
}

export default Contributions;
