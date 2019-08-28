import React, { useState } from "react";
import "./App.css";

class Header extends React.Component {
  render() {
    return (
      <div className="header">
        <div className="header__name">Captum Insights</div>
        <div className="header__nav">
          <ul>
            <li className="header__nav__item header__nav__item--active">
              Instance Attribution
            </li>
            <li className="header__nav__item">Direct Target</li>
            <li className="header__nav__item">Export</li>
          </ul>
        </div>
      </div>
    );
  }
}

class Filter extends React.Component {
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
    console.log("submitted!");
    fetch("/fetch", { method: "POST", body: JSON.stringify(this.state) });
    event.preventDefault();
  };

  render() {
    return (
      <form onSubmit={this.handleSubmit}>
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
                value={this.state.instance_type}
                onChange={this.handleInputChange}
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
                value={this.state.approximation_steps}
                onChange={this.handleInputChange}
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

function ImageFeature(props) {
  return (
    <>
      <div className="panel__column__title">Image</div>
      <div className="panel__column__body">
        <div className="gallery">
          <div className="gallery__item">
            <div className="gallery__item__image">
              <img
                src={"data:image/png;base64," + props.data.feature_output.base}
              />
            </div>
            <div className="gallery__item__description">Original</div>
          </div>
          <div className="gallery__item">
            <div className="gallery__item__image">
              <img
                src={
                  "data:image/png;base64," + props.data.feature_output.modified
                }
              />
            </div>
            <div className="gallery__item__description">Gradient Overlay</div>
          </div>
        </div>
      </div>
    </>
  );
}

class Visualization extends React.Component {
  render() {
    const v = this.props.data;
    let feature = null;
    switch (v.feature_output.type) {
      case "image":
        feature = <ImageFeature data={v} />;
        break;

      default:
        throw new Error(
          "Unsupported feature visualization type: " + v.feature_output.type
        );
    }
    return (
      <div className="panel panel--long">
        <div className="panel__column">
          <div className="panel__column__title">Predicted</div>
          <div className="panel__column__body">
            <div className="row row--padding">
              <div className="btn btn--solid">{v.actual} (0.68)</div>
            </div>
            <div className="row row--padding">
              <div className="btn btn--outline">Outdoor (0.53)</div>
            </div>
            <div className="row row--padding">
              <div className="btn btn--outline">Shopping (0.74)</div>
            </div>
          </div>
        </div>
        <div className="panel__column">
          <div className="panel__column__title">Label</div>
          <div className="panel__column__body">
            <div className="row row--padding">
              <div className="btn btn--outline">Outdoor (0.53)</div>
            </div>
            <div className="row row--padding">
              <div className="btn btn--outline">Shopping (0.74)</div>
            </div>
          </div>
        </div>
        <div className="panel__column">
          <div className="panel__column__title">Contribution</div>
          <div className="panel__column__body">
            <div className="bar-chart">
              <div className="bar-chart__group">
                <div className="bar-chart__group__bar" data-percent="50" />
                <div className="bar-chart__group__title">
                  Text{" "}
                  <span className="bar-chart__group__title__percent">88%</span>
                </div>
              </div>
              <div className="bar-chart__group">
                <div
                  className="bar-chart__group__bar width-50"
                  data-percent="25"
                />
                <div className="bar-chart__group__title">
                  Dense{" "}
                  <span className="bar-chart__group__title__percent">50%</span>
                </div>
              </div>
              <div className="bar-chart__group">
                <div
                  className="bar-chart__group__bar width-25"
                  data-percent="23"
                />
                <div className="bar-chart__group__title">
                  Image{" "}
                  <span className="bar-chart__group__title__percent">10%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="panel__column panel__column--stretch">{feature}</div>
      </div>
    );
  }
}

function Visualizations(props) {
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
      instance_type: "all",
      approximation_steps: 50
    };
  }

  render() {
    return (
      <div className="app">
        <Header />
        <Filter />
        <Visualizations data={data} />
      </div>
    );
  }
}

const data = [
  {
    actual: "cat",
    feature_output: {
      base:
        "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",
      modified:
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAAy5JREFUWIWtl81vJDUQxX9VtjuZzwQEAzmgCCkrBcEKBBxAaK/895yAA0eQclgJhBBLEjLjLg5tT3s63Un3sE/yTLtddj2/ssttAQwAhBbGNEgqNrlv7tUZCKAunjXV6dRdstl1+o2HP6xmLpKclDMqSWiycYX9NMcZSYE8YHYiQCzqPj1vC2ddglPD1kCbv8w+RyTLmx24RKiG07Nkm+pHxL1EZw0kPljxHGkUqEEC2JZj5R5BoCQSaGeYQ8LzzkXBxhPU/tflwqtSyaF5GmG+Hu0cnlTAsV+E7gzi76lHWrAWH3c7AioiBykI8bSSO2DLxy8u98RCtXhrzvcEDBAB5xyNzDWN7Le8/9EV33z+RTKvmc3OnxlSQBT11TgGIYS8j8x7b4imejAQA+zd8/dam2ph1ely39Ytl1fXJup62/qKmrVLwKmC87Q5oMGff/2RBcOoefnq+0ayDtQH7u9uwXqW1QA0GwfnCCHA7oEm/pFmK+bxPIigqlx8uGG+PufwAIM67nh98xs2YRviXCP5SVXZYrEo5PEGpZRiPlR2cfnCQnXSvOuRWqQ/NIPFuWaQ+enMlsvVgcPDODfPP/xyMxh/EFMfjAkk9nmg8p75YkWMW/5+86bQKJmIA4uI81jc9cspKXtazqDPQ733qAjeKSdBWa8Wj53Dfu8POkeac8LiCOftd4fGugaB2oy7u1tiXTqYcsqNtQ2UJ6haXSOi1AbR4J/b+9Eul+t3JhCEvg8XBRBRdjFyd/8vD9t42DiAzeaC2WyOc/4Jqy6M9lQt3nrvTVVNRG21andC3qKvvvt2cBV7Nz7rDRRMUgHsbL02TdtosVz+38HL7dafiilqTpXd9gHnHc45rj/9bIK8wxg48/c4YHRS+SYxLVc2m88NsOAmZreRs98rUGK73SHAB5sNX371dfMuTtmOhzPPMxyC7n8SagOncH11yctPro52nK8suS6d9gwPj7/0Yg2vb37lx59/6h18aEb5ZgFlqmnrff0fjZevIgqICrE+dNe9pHUJdG2scNJH/rkFOoh8VSkHGbreakGqS/6pZPckhj5L+yQWQAc8Ha3A28J/qDqd/Np/idwAAAAASUVORK5CYII=",
      type: "image"
    },
    predicted: "cat"
  },
  {
    actual: "ship",
    feature_output: {
      base:
        "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",
      modified:
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAABJZJREFUWIXdl9tu5MYRhr+qPpAczuiw8ipG1jDsbBD4wnCc17BfKs9pIL6xgYWx0npG0kjkkOzKBcnWjKS1BW+wBlJAozndVdV/Vf3V5Ahg/I6ICGaGc46UEgBmh2aqioggIgzD8Gj/feKdc9lQRACyg9mZqgKQUkJV894MZgbpvacoCpqmyQDMjJQSKSW89/R9j6pmWz87BQghZGXnHM45uq4DyNHPTuYDvPcZVNd1lEWR/ex2u2w36+wHPAwDamYH6ZqVgHzQ/kgpHWTsHvAYVdM2vP7b59zd3aGqOYMpJYZhyJGrKqqKV9WMynufS7Cf2jnS/fqa2YFe32u2bdsmr6eUCCFkEDOo2a/u13zf+Uy8lFKe54wsl0tCCHz66V/2SDg6LILnp5/fHGSw67qc5TkDzrlxTURMVU1ErSxKAyyEYICpUxMRc86Zc85evfrMAHPO2Wq1srpeGGMXmXc62gjmVEwE896Zc2rOOSuKwkII5r03ETERGfUBVsslRRGo64rVapXrbGnkQ/CBYRj4/rvvc5pDCNze3t33kwgiY2RF9JiNfAoh4L2nLAu6rsv8qqpqzJz3DlWwZJydHXN+/oIYAzFGnBvbSlSo65rt7U0+r1osDsjb9yOoZb1AxE2EHlPeti1VWXB+/skef/qxFCmlfzdNyzAMrNdrur4jJcOQTJy2bUkpsd6s2aw31HXNt//6lsvLS7xz/P31l9RVxaIqGUk9dkfXDwzDWPOmbXFO2W7H7lgsFux2O4QHN6EAOvVuCGMmbm5uiDFmMv3zm6+5vr5hUS+4eHuBiOCcIgLr9Zqm3TH0A+nBZagqpGkxxjhmQWQkUFlE884ZYCfHR5lcdV1bWRa2XNZ5bR5nL04sBJ+J+fKTU6uqaM6pBe9MRCwGbyoyknMiKpBJ6GVK9XJR0feJq+2WGEJG3TQNZy9Oubq+4qFcvlvn52EYuL65xTvP+ctjVJWLi1+nd4RiGGbkkpjZeBElM3Zdz8Wvm1yCzdUmO15UJevNht2uewQgp1aEEAL1oubq+prPT045Wq1Yr69o2payLPHecXfbHNjlF9vDoYo5VStitFd/PX+0z9TvcUr/6y++sOOjI6sXC5MndJ8+YyzBIxK+T2LwrJYLNldbYgwUMRKj55e37x4z+RkSvKPrh+fZLqqKWERiEdms78uhKrmms4iMZZx+YRjRO/rBMAzvHbtdP+1yD8B7R1WWnL04pqoq3rx5C6I4p3z91T/44T8/cnp2yl1zS9/1bLe3BOdo25amaUlmvDo/4exkybv1LZvtHe2uY1kGQnD89MvIKxUO2vMegHN4p3inxOipy5Jt03LX7BCBXTeMLxwz+pSwZNjkUEXoHzb9MyUDEMAp9OnBxh8QDwzPtH/ynKcWdW9O02zTQft2szwX/IcE+j8R/X2V/wMAelCcjwRAcfk5PVVluZ/+EAcEj+MIIWJ09Lw79Ex6n+moNX1zfiAJw3RQAGpgCzS/aQGgbizKMKQPKYFO4yVj1JfPOnwM2XLUH78Np5tTVen74eO3oYggqjCB8B8bwPg1cP/H9U+/Cf8LfmKwz0f+LgkAAAAASUVORK5CYII=",
      type: "image"
    },
    predicted: "ship"
  }
];
export default App;
