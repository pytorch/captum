import React from "react";
import Tippy from "@tippyjs/react";
import "./Tooltip.css";
import cx from "../../utils/cx";

// TODO merge with CNN inspector tooltip

export function Tooltip(props) {
  const [visible, setVisible] = React.useState(false);
  const { trigger = "mouseenter", placement = "top", delay = 0 } = props;

  return (
    <>
      <Tippy
        onMount={() => {
          setVisible(true);
          props.onMount && props.onMount();
        }}
        onHide={() => {
          setVisible(false);
          props.onHide && props.onHide();
        }}
        render={(attrs) => (
          <div
            className={cx(["popup-wrapper", visible ? "open" : ""])}
            {...attrs}
          >
            <div className="popup-content">{props.content}</div>
            <div className="popup-arrow" data-popper-arrow=""></div>
          </div>
        )}
        interactive={true}
        hideOnClick={false}
        trigger={trigger}
        placement={placement}
        delay={delay}
      >
        {props.children}
      </Tippy>
    </>
  );
}
