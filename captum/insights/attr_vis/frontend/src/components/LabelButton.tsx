import React from "react";
import cx from "../utils/cx";
import styles from "../App.module.css";

interface LabelButtonProps {
  labelIndex: number;
  inputIndex: number;
  modelIndex: number;
  active: boolean;
  onTargetClick: (
    labelIndex: number,
    inputIndex: number,
    modelIndex: number
  ) => void;
}

function LabelButton(props: React.PropsWithChildren<LabelButtonProps>) {
  const onClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    props.onTargetClick(props.labelIndex, props.inputIndex, props.modelIndex);
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
