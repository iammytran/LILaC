const fs = require('fs/promises');
const path = require('path');
const { JSDOM } = require("jsdom");

const ROOT_DIR = "mineru_ouputs";
const OUTPUT_DIR = "datasets/InfoVQA/parsed_documents/dev"

const window = new JSDOM("").window;
const DOMParser = window.DOMParser;

/* =========================
   Table Parser
========================= */

function convertHtmlTableToJson(htmlText) {

    const parser = new DOMParser();

    const doc = parser.parseFromString(htmlText, 'text/html');

    const tableElement = doc.querySelector('table');

    if (!tableElement) return null;

    const rows = Array.from(tableElement.querySelectorAll('tr'))
        .map(row =>
            Array.from(row.querySelectorAll('td'))
                .map(td => td.textContent.trim())
        )
        .filter(row => row.length > 0);

    return {
        table_caption: "Placeholder",
        refs: [],
        columns: rows[0] || [],
        table: rows
    };
}

function hasTextType(content) {

    for (const value of Object.values(content)) {

        if (!Array.isArray(value)) {
            continue;
        }

        for (const item of value) {

            if (item.type === "text") {
                return true;
            }
        }
    }

    return false;
}

function extractAllTexts(content) {
    allTexts = []

    for (const value of Object.values(content)) {

        if (!Array.isArray(value)) {
            continue;
        }

        for (const item of value) {
            if (item.type === "text") {
                allTexts.push(item.content);
                return allTexts.length > 1
                        ? allTexts.join(" ")
                        : allTexts[0];
            }
        }
    }

    return '';
}


/* =========================
   Block Processing
========================= */

function processBlocks(blocks, state) {

    blocks.forEach(item => {

        if (item.type === 'image') {

            state.imageIndex++;

            state.images[`i_${state.imageIndex}`] = {
                content: "",
                bbox: item.bbox,
                metadata: ""
            };
        }

        if (hasTextType(item.content)) {

            state.textIndex++;

            // state.texts[`p_${state.textIndex}`] = {
            //     text: item.content?.title_content?.[0]?.content || item.content?.page_header_content?.[0]?.content || item.content?.paragraph_content?.[0]?.content,
            //     edges: [],
            // };

            state.texts[`p_${state.textIndex}`] = {
                text: extractAllTexts(item.content),
                edges: [],
            };
        }

        if (item.type === 'table') {

            const htmlTable =
                item.content?.html;

            if (!htmlTable) return;

            state.tableIndex++;

            state.tables[`t_${state.tableIndex}`] =
                convertHtmlTableToJson(htmlTable);
        }
    });
}



/* =========================
   Process One File
========================= */

async function processMineruOutput(filePath) {

    const jsonString = await fs.readFile(filePath, 'utf8');

    const data = JSON.parse(jsonString);

    const state = {
        images: {},
        texts: {},
        tables: {},

        imageIndex: 0,
        textIndex: 0,
        tableIndex: 0
    };

    // TEMPORARY
    // const middleFile = data.pdf_info?.[0];
    
    // if (!middleFile) {
    //     throw new Error("pdf_info not found");
    // }

    const contentListV2File = data[0];


    // Transform text and table
    processBlocks(contentListV2File || [], state);

    // processBlocks(middleFile.para_blocks || [], state);
    // processBlocks(middleFile.discarded_blocks || [], state);

    return {
        title: `${path.basename(filePath).split('_')[0]}.jpeg`,
        url: "placeholder",

        hierarchy: "{}",

        text: state.texts,
        sentence: {},

        table: state.tables,
        table_segment: {},

        image: state.images
    };
}

/* =========================
   Main
========================= */

async function main() {

    const folders = await fs.readdir(ROOT_DIR);

    for (const folder of folders) {

        const folderPath = path.join(ROOT_DIR, folder);

        const stat = await fs.stat(folderPath);

        if (!stat.isDirectory()) {
            continue;
        }

        const mineru_output = path.join(
            folderPath,
            `${folder}_content_list_v2.json`
        );

        try {

            console.log(`Processing ${mineru_output}`);

            const output = await processMineruOutput(mineru_output);

            const outputPath = path.join(
                OUTPUT_DIR,
                `${folder}.json`
            );

            await fs.writeFile(
                outputPath,
                JSON.stringify(output, null, 2)
            );

            console.log(`Saved -> ${outputPath}`);

        } catch (err) {

            console.error(
                `Failed processing ${mineru_output}`,
                err.message
            );
        }
    }
}

main().catch(console.error);