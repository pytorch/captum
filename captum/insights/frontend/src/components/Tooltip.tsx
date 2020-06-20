import React from "react";

import styles from "../App.module.css";

function Tooltip(props: { label: string }) {
  return (
    <div className={styles.tooltip}>
      <div className={styles["tooltip__label"]}>{props.label}</div>
    </div>
  );
}

export default Tooltip;
