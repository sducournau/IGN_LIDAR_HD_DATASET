import type { ReactNode } from "react";
import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  Svg?: React.ComponentType<React.ComponentProps<"svg">>;
  imgSrc?: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: "⚡ Smart Skip Detection",
    imgSrc: "/img/indor.png",
    description: (
      <>
        Automatically skip existing downloads, enriched files, and patches.
        Resume interrupted workflows without reprocessing existing data.
      </>
    ),
  },
  {
    title: "🏗️ Multi-Level Classification",
    imgSrc: "/img/lod3.png",
    description: (
      <>
        Support for LOD2 (15 classes) and LOD3 (30 classes) building
        classification taxonomies with rich geometric features.
      </>
    ),
  },
  {
    title: "🚀 GPU Acceleration",
    imgSrc: "/img/ext.png",
    description: (
      <>
        Optional GPU support for faster feature computation using CUDA and cupy
        for large-scale LiDAR processing.
      </>
    ),
  },
];

function Feature({ title, Svg, imgSrc, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        {Svg ? (
          <Svg className={styles.featureSvg} role="img" />
        ) : (
          <img
            src={imgSrc}
            className={styles.featureSvg}
            role="img"
            alt={title}
            style={{ maxHeight: "200px", width: "auto" }}
          />
        )}
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
