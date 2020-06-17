import React from "react";
import styles from "../App.module.css";
import cx from "../utils/cx";

function NumberArgument(props) {
  var min = props.limit[0];
  var max = props.limit[1];
  return (
    <div>
      {props.name}:
      <input
        className={cx([styles.input, styles["input--narrow"]])}
        name={props.name}
        type="number"
        value={props.value}
        min={min}
        max={max}
        onChange={props.handleInputChange}
      />
    </div>
  );
}

function EnumArgument(props) {
  const options = props.limit.map((item, key) => (
    <option value={item}>{item}</option>
  ));
  return (
    <div>
      {props.name}:
      <select
        className={styles.select}
        name={props.name}
        value={props.value}
        onChange={props.handleInputChange}
      >
        {options}
      </select>
    </div>
  );
}

function StringArgument(props) {
  return (
    <div>
      {props.name}:
      <input
        className={cx([styles.input, styles["input--narrow"]])}
        name={props.name}
        type="text"
        value={props.value}
        onChange={props.handleInputChange}
      />
    </div>
  );
}

export { StringArgument, EnumArgument, NumberArgument };
