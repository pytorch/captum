import React from "react";
import cx from "../utils/cx";
import styles from "../App.module.css";

function LabelButton(props) {
  const onClick = (e) => {
    e.preventDefault();
    props.onTargetClick(props.labelIndex, props.instance);
  };

  return (
    <button
      onClick={onClick}
      className={cx({
        [styles.btn]: true,
        [styles["btn--solid"]]: props.active,
        [styles["btn--outline"]]: !props.active,
      })}
    >
      {props.children}
    </button>
  );
}

export default LabelButton;
