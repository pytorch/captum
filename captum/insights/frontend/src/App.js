import React from "react";
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
      <header className="header">
        <div className="header__name">Captum Insights</div>
        <nav className="header__nav">
          <ul>
            <li className="header__nav__item header__nav__item--active">
              Instance Attribution
            </li>
            <li className="header__nav__item">Direct Target</li>
            <li className="header__nav__item">Export</li>
          </ul>
        </nav>
      </header>
    );
  }
}

function Tooltip(props) {
  return (
    <div className="tooltip">
      <div className="tooltip__label">{props.label}</div>
    </div>
  );
}

class FilterContainer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      instance_type: "all",
      approximation_steps: 50
    };
  }

  handleInputChange = event => {
    const target = event.target;
    const value = target.type === "checkbox" ? target.checked : target.value;
    const name = target.name;
    this.setState({
      [name]: value
    });
  };

  handleSubmit = event => {
    this.props.fetchData(this.state);
    event.preventDefault();
  };

  render() {
    return (
      <Filter
        instanceType={this.state.instance_type}
        approximationSteps={this.state.approximation_steps}
        onHandleInputChange={this.handleInputChange}
        handleSubmit={this.handleSubmit}
      />
    );
  }
}

class Filter extends React.Component {
  render() {
    return (
      <form onSubmit={this.props.handleSubmit}>
        <div className="filter-panel">
          <div className="filter-panel__column">
            <div className="filter-panel__column__title">Filter by Classes</div>
            <div className="filter-panel__column__body">
              Animal and 2 other classes are selected. <a href="">Edit</a>
            </div>
          </div>
          <div className="filter-panel__column">
            <div className="filter-panel__column__title">
              Filter by Instances
            </div>
            <div className="filter-panel__column__body">
              Instance Type:{" "}
              <select
                className="select"
                name="instance_type"
                value={this.props.isntanceType}
                onChange={this.props.handleInputChange}
              >
                <option value="all">All</option>
                <option value="false_negative">False Negative</option>
                <option value="false_positive">False Positive</option>
              </select>
            </div>
          </div>
          <div className="filter-panel__column">
            <div className="filter-panel__column__title">
              Integrated Gradients
            </div>
            <div className="filter-panel__column__body">
              Approximation steps:{" "}
              <input
                className="input"
                name="approximation_steps"
                type="number"
                value={this.props.approximationSteps}
                onChange={this.props.handleInputChange}
              />
            </div>
          </div>
          <div className="filter-panel__column filter-panel__column--end">
            <button className="btn btn--outline btn--large">Fetch</button>
          </div>
        </div>
      </form>
    );
  }
}

// TODO maybe linear interpolate the colors instead of hardcoding them
function getPercentageColor(percentage, zeroDefault = false) {
  if (percentage > 50) {
    return "percentage-blue";
  } else if (percentage > 10) {
    return "percentage-light-blue";
  } else if (percentage > -10) {
    if (zeroDefault) {
      return "percentage-white";
    }
    return "percentage-gray";
  } else if (percentage > -50) {
    return "percentage-light-red";
  } else {
    return "percentage-red";
  }
}

function ImageFeature(props) {
  return (
    <>
      <div className="panel__column__title">{props.data.name} (Image)</div>
      <div className="panel__column__body">
        <div className="gallery">
          <div className="gallery__item">
            <div className="gallery__item__image">
              <img src={"data:image/png;base64," + props.data.base} />
            </div>
            <div className="gallery__item__description">Original</div>
          </div>
          <div className="gallery__item">
            <div className="gallery__item__image">
              <img src={"data:image/png;base64," + props.data.modified} />
            </div>
            <div className="gallery__item__description">Gradient Overlay</div>
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
          className={cx([
            getPercentageColor(props.data.modified[i], /* zeroDefault */ true),
            "text-feature-word"
          ])}
        >
          {w}
          <Tooltip label={props.data.modified[i]} />
        </span>{" "}
      </>
    );
  });
  return (
    <>
      <div className="panel__column__title">{props.data.name} (Text)</div>
      <div className="panel__column__body">{color_words}</div>
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
    default:
      throw new Error("Unsupported feature visualization type: " + data.type);
  }
}

class Contributions extends React.Component {
  render() {
    return this.props.feature_outputs.map(f => (
      <div className="bar-chart__group">
        <div
          className={cx({
            "bar-chart__group__bar": true,
            [getPercentageColor(f.contribution)]: true
          })}
          width={f.contribution + "%"}
        />
        <div className="bar-chart__group__title">{f.name}</div>
      </div>
    ));
  }
}

class Visualization extends React.Component {
  render() {
    const data = this.props.data;
    const features = data.feature_outputs.map(f => <Feature data={f} />);

    return (
      <div className="panel panel--long">
        <div className="panel__column">
          <div className="panel__column__title">Predicted</div>
          <div className="panel__column__body">
            {data.predicted.map((p, i) => (
              <div className="row row--padding">
                <div
                  className={cx({
                    btn: true,
                    "btn--solid": i === 0,
                    "btn--outline": i !== 0
                  })}
                >
                  {p.label} ({p.score.toFixed(3)})
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="panel__column">
          <div className="panel__column__title">Label</div>
          <div className="panel__column__body">
            <div className="row row--padding">
              <div className="btn btn--outline">{data.actual}</div>
            </div>
          </div>
        </div>
        <div className="panel__column">
          <div className="panel__column__title">Contribution</div>
          <div className="panel__column__body">
            <div className="bar-chart">
              <Contributions feature_outputs={data.feature_outputs} />
            </div>
          </div>
        </div>
        <div className="panel__column panel__column--stretch">{features}</div>
      </div>
    );
  }
}

function Visualizations(props) {
  if (!props.data || props.data.length === 0) {
    return (
      <div className="viz">
        <div className="panel">
          <div className="filter-panel__column">
            Please press <strong className="text-feature-word">Fetch</strong> to
            start loading data.
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="viz">
      {props.data.map((v, i) => (
        <Visualization data={v} key={i} />
      ))}
    </div>
  );
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: []
    };
  }

  fetchData = filter_config => {
    console.log("filter config: ", filter_config);
    fetch("/fetch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(filter_config)
    })
      .then(response => response.json())
      .then(response => this.setState({ data: response }));
  };

  render() {
    return (
      <div className="app">
        <Header />
        <FilterContainer fetchData={this.fetchData} />
        <Visualizations data={this.state.data} />
      </div>
    );
  }
}

export default App;
