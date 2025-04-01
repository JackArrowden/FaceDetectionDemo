import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import './index.css'

import Home from "./components/landing/Home";
import DemoPage from "./components/demo/Demo";
import AboutPage from "./components/about/About";

const router = createBrowserRouter([
    {
        path: "/",
        element: <Home />,
    },
    {
        path: "/demo",
        element: <DemoPage />,
    },
    {
        path: "/about",
        element: <AboutPage />,
    }
]);

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <RouterProvider router={router} />
    </StrictMode>,
)