import React from "react";
import styles from "../App.module.css";
import cx from "../utils/cx";

function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.header__name}>Captum Insights</div>
      <nav className={styles.header__nav}>
        <ul>
          <li
            className={cx([
              styles.header__nav__item,
              styles["header__nav__item--active"],
            ])}
          >
            Instance Attribution
          </li>
        </ul>
      </nav>
    </header>
  );
}

export default Header;
