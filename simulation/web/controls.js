const visualizationElements = [];
const elements = visualizationElements;
const stepCounter = document.getElementById("step-counter");
const fpsSlider = document.getElementById("fps-slider");
const fpsValue = document.getElementById("fps-value");
const startStopButton = document.getElementById("play-pause");

function ModelController() {
    this.tick = 0;

    this.running = false;
    this.finished = false;
    this.fps = 10;
    this.interval = null;

    this.receivedFrames = 0;

    this.step = () => {
        this.tick += 1;
        stepCounter.innerText = this.tick;
        send({ type: "get_step", step: this.tick });
        return this.tick;
    }

    this.render = (data) => {
        this.receivedFrames += 1;
        console.log("Received frames " + this.receivedFrames + " / " + this.tick)
        if (this.tick - this.receivedFrames > 10) {
            console.log("Too many frames behind, pausing");
            this.stop();
            startStopButton.classList.add("play");
        }
        else if (this.tick - this.receivedFrames > 2) {
            console.log("Too many frames behind, reducing FPS");
            this.changeFPS(Math.ceil(this.fps * 0.8));
        } 
        else if (fpsSlider.value - this.fps > 2) {
            console.log("Attempting to increase FPS")
            this.changeFPS(Math.ceil(this.fps + 1));
        }

        // console.log(Number(fpsValue.innerText) - this.fps);

        visualizationElements.forEach((element, index) => {
            element.render(data[index]);
        });
    }

    this.done = () => {
        this.stop();
        startStopButton.classList.add("play")
        startStopButton.classList.add("finished")
        startStopButton.classList.add("disabled")

        this.finished = true;
    }

    this.reset = () => {
        this.tick = 0;
        this.receivedFrames = 0;
        startStopButton.classList.remove("disabled")
        startStopButton.classList.remove("finished")
        this.finished = false;
        // this.changeFPS(fpsSlider.value);
        visualizationElements.forEach((element) => {
            element.reset()
        });
        send({ type: "reset" });
        
        // Delay a step
        // setTimeout(() => {
        //     this.step();
        // }, 100);
        return this.tick;
    }

    this.start = () => {
        this.running = true;
        this.interval = setInterval(() => {
            controller.step();
        }, 1000 / this.fps);
    }

    this.stop = () => {
        this.running = false;
        clearInterval(this.interval);
    }

    this.changeFPS = (fps) => {
        this.fps = fps;
        if (fps < fpsSlider.value) {
            fpsValue.innerText = fps + " (set: " + fpsSlider.value + ")";
        } else {
            fpsValue.innerText = fps;
        }
        if (this.running) {
            this.stop();
            this.start();
        }
    }
}

const controller = new ModelController();

const changeInnerText = (element, text) => {
    element.firstElementChild.innerText = text;
}

// Play/Pause button
startStopButton.addEventListener("click", () => {
    if (controller.finished) { return }
    if (controller.running) {
        controller.stop();
        startStopButton.classList.add("play")
    } else {
        controller.start();
        startStopButton.classList.remove("play")
    }
});

// Step button
const stepButton = document.getElementById("step");
stepButton.addEventListener("click", () => {
    if (controller.finished) { return }
    controller.step();
});

// Reset button
const resetButton = document.getElementById("reset");
resetButton.addEventListener("click", () => {
    controller.reset();
});

// Save params button
const saveParamsButton = document.getElementById("save-params");
saveParamsButton.addEventListener("click", () => {
    controller.reset();
});

// FPS slider
fpsSlider.addEventListener("input", () => {
    fpsValue.innerText = fpsSlider.value;
    controller.changeFPS(fpsSlider.value);
});

// Websocket connection
const ws = new WebSocket(
    (window.location.protocol === "https:" ? "wss://" : "ws://") +
    location.host +
    "/ws"
);

ws.onmessage = function (message) {
    const msg = JSON.parse(message.data);
    switch (msg["type"]) {
        case "viz_state":
            // Update visualization state
            controller.render(msg["data"]);
            break;
        case "end":
            // We have reached the end of the model
            controller.done();
            break;
        case "model_params":
            // Create GUI elements for each model parameter and reset everything
            initGUI(msg["params"]);
            controller.reset();
            break;
        default:
            // There shouldn't be any other message
            console.log("Unexpected message.");
            console.log(msg);
    }
};

const send = function (message) {
    const msg = JSON.stringify(message);
    ws.send(msg);
};


// Options pane

const modelParams = document.getElementById("model-params");
const visOptions = document.getElementById("vis-options");

const addBooleanInput = (key, obj, callback) => {
    console.log(key, obj);
    const div = document.createElement("div");
    div.classList.add("option");
    div.innerHTML = `
        <span class="label">${obj.name}</span>
        <input type="checkbox" name="${key}" id="${key}" ${obj.value ? "checked" : ""}>
    `;
    div.addEventListener("change", (e) => {
        callback(key, e.target.checked);
    });
    return div
}

const addSliderInput = (key, obj, callback) => {
    const div = document.createElement("div");
    div.classList.add("option");
    div.innerHTML = `
        <span class="label">${obj.name}<span class="value-label"></span></span>
        <input type="range" name="${key}" id="${key}" min="${obj.min_value}" max="${obj.max_value}", step="${obj.step}" value="${obj.value}">
    `;
    let valueLabel = div.querySelector(".value-label");
    div.addEventListener("change", (e) => {
        callback(key, Number(e.target.value));
    });
    div.addEventListener("input", (e) => {
        valueLabel.innerText = e.target.value;
    });
    valueLabel.innerText = obj.value;
    return div
}

const addNumberInput = (key, obj, callback) => {
    const div = document.createElement("div");
    div.classList.add("option");
    div.innerHTML = `
        <span class="label">${obj.name}</span>
        <input type="number" name="${key}" id="${key}" min="${obj.min_value}" max="${obj.max_value}", step="${obj.step}" value="${obj.value}">
    `;
    div.addEventListener("change", (e) => {
        callback(key, Number(e.target.value));
    });
    return div
}

const addChoiceInput = (key, obj, callback) => {
    const div = document.createElement("div");
    div.classList.add("option");
    div.innerHTML = `
        <span class="label">${obj.name}</span>
        <select name="${key}" id="${key}">
        </select>
    `;
    const select = div.querySelector("select");
    for (const option of obj.choices) {
        const optionElement = document.createElement("option");
        optionElement.value = option;
        optionElement.innerText = option;
        select.appendChild(optionElement);
    }
    select.value = obj.value;
    select.addEventListener("change", (e) => {
        callback(key, e.target.value);
    });
    return div
}

const addObjectInput = (key, obj, callback) => {
    // Uses a textarea to allow for JSON input
    let lastContent = "";
    const div = document.createElement("div");
    div.classList.add("option");
    div.classList.add("option-vertical");
    div.innerHTML = `
        <span class="label">${obj.name}</span>
        <div 
            class="json-input"
            contenteditable
            data-gramm="false"
            spellcheck="false"
            style="width: 100%; min-height: 100%;"
            name="${key}"
            id="${key}"
        ></div>
    `;
    const textarea = div.querySelector("div.json-input");
    textarea.innerText = JSON.stringify(JSON.parse(obj.value), null, 2);
    textarea.addEventListener("focusout", (e) => {
        stripped = e.target.innerText.replace(/\s/g, "");
        // Check if the input is valid JSON
        if (stripped === lastContent) {
            return;
        }
        try {
            JSON.parse(stripped);
        } catch (e) {
            return;
        }
        lastContent = stripped;
        callback(key, stripped);
    });
    textarea.addEventListener("keydown", (e) => {
        // Get the location of the cursor
        const selection = window.getSelection();
        const range = selection.getRangeAt(0);

        if (e.key === "Tab") {
            e.preventDefault();
            document.execCommand("insertHTML", false, "\u00a0\u00a0");
        }
        if (e.key === "Enter") {
            const text = range.startContainer.data;
            const line = text.substring(0, range.startOffset);
            const indent = line.match(/^\s+/);
            if (indent) {
                e.preventDefault();
                document.execCommand("insertHTML", false, "\n" + indent[0]);
            }
        }
    });
    return div
}

// const addObject2Input = (key, obj, callback) => {
//     // Renders the object using 
//     const value = JSON.parse(obj.value);
//     const formatter = new JSONFormatter(value, 1, {
//         hoverPreviewEnabled: false,
//         hoverPreviewArrayCount: 100,
//         hoverPreviewFieldCount: 5,
//         theme: 'dark',
//         animateOpen: true,
//         animateClose: false,
//         useToJSON: true,
//         maxArrayItems: 100,
//         exposePath: true
//     });
//     formatter.openAtDepth(3);

//     const div = document.createElement("div");
//     div.classList.add("option");
//     div.classList.add("option-vertical");
//     div.innerHTML = `
//         <span class="label">${obj.name}</span>
//     `;
//     div.appendChild(formatter.render());
//     return div
// }


const initGUI = (paramsMessage) => {
    const onSubmitCallback = (paramName, value) => {
        send({ type: "submit_params", param: paramName, value: value });
    }

    const addParamInput = (paramName, obj) => {
        switch (obj.param_type) {
            case "checkbox":
                modelParams.appendChild(addBooleanInput(paramName, obj, onSubmitCallback));
                break;
            case "slider":
                modelParams.appendChild(addSliderInput(paramName, obj, onSubmitCallback));
                break;
            case "number":
                modelParams.appendChild(addNumberInput(paramName, obj, onSubmitCallback));
                break;
            case "choice":
                modelParams.appendChild(addChoiceInput(paramName, obj, onSubmitCallback));
                break;
            case "object":
                modelParams.appendChild(addObjectInput(paramName, obj, onSubmitCallback));
                break;
        }
    }

    for (const option in paramsMessage) {
        const paramStr = String(option);
        addParamInput(paramStr, paramsMessage[option]); // catch-all for params that use Option class
    }
}