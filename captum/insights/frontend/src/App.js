import React from "react";
import ReactTags from "react-tag-autocomplete";
import styles from "./App.module.css";
import "./App.css";

// helper method to convert an array or object into a valid classname
function cx(obj) {
  if (Array.isArray(obj)) {
    return obj.join(" ");
  }
  return Object.keys(obj)
    .filter(k => !!obj[k])
    .join(" ");
}

class Header extends React.Component {
  render() {
    return (
      <header className={styles.header}>
        <div className={styles.header__name}>Captum Insights</div>
        <nav className={styles.header__nav}>
          <ul>
            <li className={cx([styles.header__nav__item, styles['header__nav__item--active']])}>
              Instance Attribution
            </li>
          </ul>
        </nav>
      </header>
    );
  }
}

function Tooltip(props) {
  return (
    <div className={styles.tooltip}>
      <div className={styles['tooltip__label']}>{props.label}</div>
    </div>
  );
}

class FilterContainer extends React.Component {
  constructor(props) {
    super(props);
    const suggested_classes = props.config.map((c, i) => ({ id: i, name: c }));

    this.state = {
      prediction: "all",
      approximation_steps: 20,
      classes: [],
      suggested_classes: suggested_classes
    };
  }

  handleClassDelete = i => {
    const classes = this.state.classes.slice(0);
    const removed_class = classes.splice(i, 1);
    const suggested_classes = [].concat(
      this.state.suggested_classes,
      removed_class
    );
    this.setState({ classes, suggested_classes });
  };

  handleClassAdd = added_class => {
    const classes = [].concat(this.state.classes, added_class);
    const suggested_classes = this.state.suggested_classes.filter(
      t => t.id !== added_class.id
    );
    this.setState({ classes, suggested_classes });
  };

  handleInputChange = event => {
    const target = event.target;
    const value = target.type === "checkbox" ? target.checked : target.value;
    const name = target.name;
    this.setState({
      [name]: value
    });
  };

  handleSubmit = event => {
    const data = {
      prediction: this.state.prediction,
      approximation_steps: this.state.approximation_steps,
      classes: this.state.classes.map(i => i["name"])
    };
    this.props.fetchData(data);
    event.preventDefault();
  };

  render() {
    return (
      <Filter
        prediction={this.state.prediction}
        classes={this.state.classes}
        suggestedClasses={this.state.suggested_classes}
        approximationSteps={this.state.approximation_steps}
        handleClassAdd={this.handleClassAdd}
        handleClassDelete={this.handleClassDelete}
        handleInputChange={this.handleInputChange}
        handleSubmit={this.handleSubmit}
      />
    );
  }
}

class ClassFilter extends React.Component {
  render() {
    return (
      <ReactTags
        tags={this.props.classes}
        autofocus={false}
        suggestions={this.props.suggestedClasses}
        handleDelete={this.props.handleClassDelete}
        handleAddition={this.props.handleClassAdd}
        minQueryLength={0}
        placeholder="add new class..."
      />
    );
  }
}

class Filter extends React.Component {
  render() {
    return (
      <form onSubmit={this.props.handleSubmit}>
        <div className={styles["filter-panel"]}>
          <div className={styles["filter-panel__column"]}>
            <div className={styles["filter-panel__column__title"]}>Filter by Classes</div>
            <div className={styles["filter-panel__column__body"]}>
              <ClassFilter
                handleClassDelete={this.props.handleClassDelete}
                handleClassAdd={this.props.handleClassAdd}
                suggestedClasses={this.props.suggestedClasses}
                classes={this.props.classes}
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
                onChange={this.props.handleInputChange}
                value={this.props.prediction}
              >
                <option value="all">All</option>
                <option value="correct">Correct</option>
                <option value="incorrect">Incorrect</option>
              </select>
            </div>
          </div>
          <div className={styles["filter-panel__column"]}>
            <div className={styles["filter-panel__column__title"]}>
              Integrated Gradients
            </div>
            <div className={styles["filter-panel__column__body"]}>
              Approximation steps:{" "}
              <input
                className={cx([styles.input, styles["input--narrow"]])}
                name="approximation_steps"
                type="number"
                value={this.props.approximationSteps}
                onChange={this.props.handleInputChange}
              />
            </div>
          </div>
          <div className={cx([styles["filter-panel__column"], styles["filter-panel__column--end"]])}>
            <button className={cx([styles.btn, styles["btn--outline"], styles["btn--large"]])}>Fetch</button>
          </div>
        </div>
      </form>
    );
  }
}

function Spinner(_) {
  return <div className={styles.spinner} />;
}

function calcHSLFromScore(percentage, zeroDefault = false) {
  const blue_hsl = [220, 100, 80];
  const red_hsl = [10, 100, 67];

  let target_hsl = null;
  if (percentage > 0) {
    target_hsl = blue_hsl;
  } else {
    target_hsl = red_hsl;
  }

  const default_hsl = [0, 40, zeroDefault ? 100 : 90];
  const abs_percent = Math.abs(percentage * 0.01);
  if (abs_percent < 0.02) {
    return default_hsl;
  }

  const color = [
    target_hsl[0],
    (target_hsl[1] - default_hsl[1]) * abs_percent + default_hsl[1],
    (target_hsl[2] - default_hsl[2]) * abs_percent + default_hsl[2]
  ];
  return `hsl(${color[0]}, ${color[1]}%, ${color[2]}%)`;
}

function ImageFeature(props) {
  return (
    <>
      <div className={styles["panel__column__title"]}>{props.data.name} (Image)</div>
      <div className={styles["panel__column__body"]}>
        <div className={styles.gallery}>
          <div className={styles["gallery__item"]}>
            <div className={styles["gallery__item__image"]}>
              <img src={"data:image/png;base64," + props.data.base} />
            </div>
            <div className={styles["gallery__item__description"]}>Original</div>
          </div>
          <div className={styles["gallery__item"]}>
            <div className={styles["gallery__item__image"]}>
              <img src={"data:image/png;base64," + props.data.modified} />
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
            backgroundColor: calcHSLFromScore(props.data.modified[i], false)
          }}
          className={"text-feature-word"}
        >
          {w}
          <Tooltip label={props.data.modified[i].toFixed(3)} />
        </span>{" "}
      </>
    );
  });
  return (
    <>
      <div className={styles["panel__column__title"]}>{props.data.name} (Text)</div>
      <div className={styles["panel__column__body"]}>{color_words}</div>
    </>
  );
}

function GeneralFeature(props) {
  const bars = props.data.base.map((w, i) => {
    const percent = props.data.modified[i].toFixed(3);
    const color =
      "general-feature__bar__" + (percent > 0 ? "positive" : "negative");
    const width_percent = Math.abs(percent) + "%";
    return (
      <div className={styles["general-feature"]}>
        <span className={styles["general-feature__label-container"]}>
          <span className={styles["general-feature__label"]}>{w}</span>
          <span className={styles["general-feature__percent"]}>{percent}</span>
        </span>
        <div className={styles["general-feature__bar-container"]}>
          <span
            className={cx([styles["general-feature__bar"], color])}
            style={{ width: width_percent }}
          ></span>
        </div>
      </div>
    );
  });
  return (
    <>
      <div className={styles["panel__column__title"]}>{props.data.name} (General)</div>
      <div className={styles["panel__column__body"]}>{bars}</div>
    </>
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

class Contributions extends React.Component {
  render() {
    return this.props.feature_outputs.map(f => {
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
              backgroundColor: calcHSLFromScore(contribution)
            }}
          />
          <div className={styles["bar-chart__group__title"]}>{f.name}</div>
        </div>
      );
    });
  }
}

class LabelButton extends React.Component {
  onClick = e => {
    e.preventDefault();
    this.props.onTargetClick(this.props.labelIndex, this.props.instance);
  };

  render() {
    return (
      <button
        onClick={this.onClick}
        className={cx({
          [styles.btn]: true,
          [styles["btn--solid"]]: this.props.active,
          [styles["btn--outline"]]: !this.props.active
        })}
      >
        {this.props.children}
      </button>
    );
  }
}

class Visualization extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: false
    };
  }

  onTargetClick = (labelIndex, instance) => {
    this.setState({ loading: true });
    this.props.onTargetClick(labelIndex, instance, () =>
      this.setState({ loading: false })
    );
  };

  render() {
    const data = this.props.data;
    const features = data.feature_outputs.map(f => <Feature data={f} />);

    return (
      <>
        {this.state.loading && (
          <div className={styles.loading}>
            <Spinner />
          </div>
        )}
        <div
          className={cx({
            [styles.panel]: true,
            [styles["panel--long"]]: true,
            [styles["panel--loading"]]: this.state.loading
          })}
        >
          <div className={styles["panel__column"]}>
            <div className={styles["panel__column__title"]}>Predicted</div>
            <div className={styles["panel__column__body"]}>
              {data.predicted.map(p => (
                <div className={cx([styles.row, styles["row--padding"]])}>
                  <LabelButton
                    onTargetClick={this.onTargetClick}
                    labelIndex={p.index}
                    instance={this.props.instance}
                    active={p.index === data.active_index}
                  >
                    {p.label} ({p.score.toFixed(3)})
                  </LabelButton>
                </div>
              ))}
            </div>
          </div>
          <div className={styles["panel__column"]}>
            <div className={styles["panel__column__title"]}>Label</div>
            <div className={styles["panel__column__body"]}>
              <div className={styles["row row--padding"]}>
                <LabelButton
                  onTargetClick={this.onTargetClick}
                  labelIndex={data.actual.index}
                  instance={this.props.instance}
                  active={data.actual.index === data.active_index}
                >
                  {data.actual.label}
                </LabelButton>
              </div>
            </div>
          </div>
          <div className={styles["panel__column"]}>
            <div className={styles["panel__column__title"]}>Contribution</div>
            <div className={styles["panel__column__body"]}>
              <div className={styles["bar-chart"]}>
                <Contributions feature_outputs={data.feature_outputs} />
              </div>
            </div>
          </div>
          <div className={cx([styles["panel__column"], styles["panel__column--stretch"]])}>{features}</div>
        </div>
      </>
    );
  }
}

function Visualizations(props) {
  if (props.loading) {
    return (
      <div className="viz">
        <div className={cx([styles.panel, styles["panel--center"]])}>
          <Spinner />
        </div>
      </div>
    );
  }

  if (!props.data || props.data.length === 0) {
    return (
      <div className={styles.viz}>
        <div className={styles.panel}>
          <div className={styles["panel__column"]}>
            Please press <strong className={styles["text-feature-word"]}>Fetch</strong> to
            start loading data.
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className={styles.viz}>
      {props.data.map((v, i) => (
        <Visualization
          data={v}
          instance={i}
          key={i}
          onTargetClick={props.onTargetClick}
        />
      ))}
    </div>
  );
}

class AppBase extends React.Component {
  componentDidMount() {
    this.props.fetchInit();
  }

  render() {
    return (
      <div className={styles.app}>
        <Header />
        <FilterContainer
          fetchData={this.props.fetchData}
          config={this.props.config}
          key={this.props.config}
        />
        <Visualizations
          data={this.props.data}
          loading={this.props.loading}
          onTargetClick={this.props.onTargetClick}
        />
      </div>
    );
  }
}

export default AppBase;
