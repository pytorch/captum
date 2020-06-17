import React from "react";

import styles from "../App.module.css";

function Tooltip(props) {
  return (
    <div className={styles.tooltip}>
      <div className={styles["tooltip__label"]}>{props.label}</div>
    </div>
  );
}

export default Tooltip;
