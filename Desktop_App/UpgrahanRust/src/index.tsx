/* @refresh reload */
import { render } from "solid-js/web";
import "./styles/main.css";
import "maplibre-gl/dist/maplibre-gl.css";
import { App } from "./app";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element.");

render(() => <App />, root);
