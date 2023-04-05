// puppeteer-extra is a drop-in replacement for puppeteer,
// it augments the installed puppeteer with plugin functionality
const puppeteer = require('puppeteer-extra');
const { spawn } = require('child_process');
const fs = require('fs');
const AdblockerPlugin = require('puppeteer-extra-plugin-adblocker')
// add stealth plugin and use defaults (all evasion techniques)
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());
puppeteer.use(AdblockerPlugin());

puppeteer.launch({
    headless: false,
    args: ['--start-maximized', '--no-sandbox']
}).then(async browser => {
    const myArgs = process.argv.slice(2);
    console.log('myArgs: ', myArgs);
    query = myArgs[0];
    const page = await browser.newPage();
    await page.goto('https://www.youtube.com/results?search_query=' + query);
    await page.waitForSelector('#video-title');
    await page.waitFor(1000)
    const distance = 100;
    const delay = 100;
    const scrollDelay = 800;
    let data = await page.evaluate(() => {
        var channels = []
        document.querySelectorAll('#channel-info > a').forEach(function (element) {
            channels.push(element.href)
        });
        return channels
    })
    var channels = data;
    console.log('channels: ', data);
    //clear channels.txt
    fs.writeFileSync('channels.txt', '');
    for (var i = 0; i < channels.length; i++) {
        currentChannel = channels[i];
        splitList = currentChannel.split('/');
        //get the last part of the url
        channelName = splitList[splitList.length - 1];
        //write channel name to file
        fs.appendFileSync('channels.txt', channelName + '\n');
    }
    const subprocess = runScript(query);
    // print output of script
    subprocess.stdout.on('data', (data) => {
        console.log(`data:${data}`);
    });
    subprocess.stderr.on('data', (data) => {
        console.log(`error:${data}`);
    });
    subprocess.stderr.on('close', () => {
        console.log("Closed");
    });
    //await page.screenshot({ path: 'example.png' });

    await browser.close();
});

function runScript(topic) {
    return spawn('venv/bin/python3', [
        'vido.py',
        `${topic}`
    ]);
}

async function autoScroll(page) {
    await page.evaluate(async () => {
        await new Promise((resolve, reject) => {
            var totalHeight = 0;
            var distance = 100;
            var timer = setInterval(() => {
                var scrollHeight = page.document.body.scrollHeight;
                window.scrollBy(0, distance);
                totalHeight += distance;

                if (totalHeight >= scrollHeight) {
                    clearInterval(timer);
                    resolve();
                }
            }, 1);
        });
    });
}