import React from "react";
import { StringArgument, EnumArgument, NumberArgument } from "./Arguments";
import cx from "../utils/cx";
import styles from "../App.module.css";
import ClassFilter from "./ClassFilter";

const ConfigType = Object.freeze({
  Number: "number",
  Enum: "enum",
  String: "string",
});

function Filter(props) {
  const createComponentFromConfig = (name, config) => {
    switch (config.type) {
      case ConfigType.Number:
        return (
          <NumberArgument
            name={name}
            limit={config.limit}
            value={config.value}
            handleInputChange={props.handleArgumentChange}
          />
        );
      case ConfigType.Enum:
        return (
          <EnumArgument
            name={name}
            limit={config.limit}
            value={config.value}
            handleInputChange={props.handleArgumentChange}
          />
        );
      case ConfigType.String:
        return (
          <StringArgument
            name={name}
            value={config.value}
            handleInputChange={props.handleArgumentChange}
          />
        );
      default:
        throw new Error("Unsupported config type: " + config.type);
    }
  };

  const methods = props.methods.map((item, key) => (
    <option key={key} value={item}>
      {item}
    </option>
  ));
  var method_args_components = null;
  if (props.selectedMethod in props.methodArguments) {
    const method_arguments = props.methodArguments[props.selectedMethod];
    method_args_components = Object.keys(method_arguments).map((key, idx) =>
      createComponentFromConfig(key, method_arguments[key])
    );
  }
  return (
    <form onSubmit={props.handleSubmit}>
      <div className={styles["filter-panel"]}>
        <div className={styles["filter-panel__column"]}>
          <div className={styles["filter-panel__column__title"]}>
            Filter by Classes
          </div>
          <div className={styles["filter-panel__column__body"]}>
            <ClassFilter
              handleClassDelete={props.handleClassDelete}
              handleClassAdd={props.handleClassAdd}
              suggestedClasses={props.suggestedClasses}
              classes={props.classes}
            />
          </div>
        </div>
        <div className={styles["filter-panel__column"]}>
          <div className={styles["filter-panel__column__title"]}>
            Filter by Instances
          </div>
          <div className={styles["filter-panel__column__body"]}>
            Prediction:{" "}
            <select
              className={styles.select}
              name="prediction"
              onChange={props.handleInputChange}
              value={props.prediction}
            >
              <option value="all">All</option>
              <option value="correct">Correct</option>
              <option value="incorrect">Incorrect</option>
            </select>
          </div>
        </div>
        <div className={styles["filter-panel__column"]}>
          <div className={styles["filter-panel__column__title"]}>
            Choose Attribution Method
          </div>
          <div className={styles["filter-panel__column__body"]}>
            Attribution Method:{" "}
            <select
              className={styles.select}
              name="selected_method"
              onChange={props.handleInputChange}
              value={props.selectedMethod}
            >
              {methods}
            </select>
          </div>
        </div>
        <div className={styles["filter-panel__column"]}>
          <div className={styles["filter-panel__column__title"]}>
            Attribution Method Arguments
          </div>
          <div className={styles["filter-panel__column__body"]}>
            {method_args_components}
          </div>
        </div>
        <div
          className={cx([
            styles["filter-panel__column"],
            styles["filter-panel__column--end"],
          ])}
        >
          <button
            className={cx([
              styles.btn,
              styles["btn--outline"],
              styles["btn--large"],
            ])}
          >
            Fetch
          </button>
        </div>
      </div>
    </form>
  );
}

export default Filter;
