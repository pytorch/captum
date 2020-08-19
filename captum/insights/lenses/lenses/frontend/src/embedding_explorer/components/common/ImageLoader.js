import React from "react";
import cx from "../../utils/cx";
import Spinner from "./Spinner";
import "./ImageLoader.css";

export function ImageLoader(props) {
  const { src = "", defaultWidth = 32, defaultHeight = 32 } = props;

  const imgRef = React.useRef();
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    setLoaded(false);
  }, [src]);

  const onLoad = (e) => {
    if (imgRef.current && imgRef.current.getAttribute("src") === src) {
      setLoaded(true);
    }
  };

  const onError = (e) => {
    if (imgRef.current && imgRef.current.getAttribute("src") === src) {
      setLoaded(true);
    }
  };

  const loaderStyle = loaded ? { opacity: 0.0 } : { opacity: 1 };

  const wrapperStyle = {};
  let width, height;
  if (imgRef.current != null) {
    width =
      imgRef.current.naturalWidth === 0
        ? defaultWidth
        : imgRef.current.naturalWidth;
    height =
      imgRef.current.naturalHeight === 0
        ? defaultHeight
        : imgRef.current.naturalHeight;
  } else {
    [width, height] = [defaultWidth, defaultHeight];
  }

  wrapperStyle.width = `${width}px`;
  wrapperStyle.height = `${height}px`;

  return (
    <div
      className={cx(["image-loader-container", props.className])}
      style={wrapperStyle}
    >
      <img
        ref={imgRef}
        src={src}
        alt={props.alt}
        onLoad={onLoad}
        onError={onError}
      />
      <div className="image-loader" style={loaderStyle}>
        <Spinner size={12} borderWidth={1} color="#656565" />
      </div>
    </div>
  );
}
