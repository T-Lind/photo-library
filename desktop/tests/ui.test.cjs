// Real browser interactions against the shipped UI, with a deterministic API.
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { createServer } = require('node:http');
const { readFileSync, mkdirSync } = require('node:fs');
const { join } = require('node:path');
const { chromium } = require('playwright');

let server, browser, origin;
const photos = Array.from({ length: 55 }, (_, image_id) => ({
  image_id, filename: `holiday-${String(image_id).padStart(3, '0')}.jpg`,
  taken_at: '2024-07-04T12:00:00', face_count: 0, media_type: 'image',
  favorite: false, rating: 0,
}));

before(async () => {
  server = createServer((req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    const files = { '/': 'index.html', '/app.js': 'app.js', '/curation.js': 'curation.js', '/app.css': 'app.css' };
    if (!files[path]) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', path.endsWith('.js') ? 'text/javascript' : path.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(readFileSync(join(__dirname, '../ui', files[path])));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true });
});

after(async () => {
  await browser?.close();
  if (server) await new Promise(resolve => server.close(resolve));
});

async function openApp(t, viewport = { width: 1280, height: 900 }) {
  const page = await browser.newPage({ viewport });
  page.setDefaultTimeout(10000);
  t.after(() => page.close());
  const errors = [], searches = [], saved = [];
  page.on('pageerror', error => errors.push(error.message));
  t.after(() => assert.deepEqual(errors, []));
  await page.route('**/api/v1/**', async route => {
    const req = route.request();
    const url = new URL(req.url());
    const path = url.pathname.slice('/api/v1'.length);
    const payload = req.method() === 'POST' ? req.postDataJSON() : null;
    let body;
    if (path === '/health') body = { ready: true };
    else if (path === '/admin/models') body = { ready: true, models: [] };
    else if (path === '/admin/jobs') body = [];
    else if (path === '/people') body = [];
    else if (path === '/stats') body = { total_images: 55, total_faces: 0, total_people: 0 };
    else if (path === '/cameras') body = { cameras: [] };
    else if (path === '/timeline') body = { months: [] };
    else if (path === '/saved-searches') {
      if (payload) saved.push({ ...payload, id: String(saved.length + 1) });
      body = payload ? saved.at(-1) : { searches: saved };
    } else if (path === '/search') {
      searches.push(payload);
      const pageSize = payload.per_page, offset = (payload.page - 1) * pageSize;
      body = { results: photos.slice(offset, offset + pageSize), total: photos.length, page: payload.page, per_page: pageSize };
    } else if (path === '/albums') body = { albums: [{ album_id: 1, name: 'Summer holiday', photo_count: 55, cover_image_id: 0 }] };
    else if (path === '/albums/1/suggestions') body = { suggestions: [] };
    else if (path === '/albums/1') {
      const pageNumber = Number(url.searchParams.get('page') || 1);
      const pageSize = Number(url.searchParams.get('per_page') || 48);
      const offset = (pageNumber - 1) * pageSize;
      body = { album_id: 1, name: 'Summer holiday', photo_count: 55, total: 55, page: pageNumber, per_page: pageSize,
        has_more: offset + pageSize < 55, images: photos.slice(offset, offset + pageSize) };
    } else if (/^\/images\/\d+\/details$/.test(path)) {
      const id = Number(path.split('/')[2]);
      body = { ...photos[id], faces: [], people: [], width: 240, height: 180 };
    } else if (/^\/images\/\d+(\/thumbnail)?$/.test(path)) {
      return route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="180"><rect width="240" height="180" fill="#375c66"/></svg>' });
    } else {
      errors.push(`Unexpected API request: ${req.method()} ${path}`);
      return route.fulfill({ status: 404, json: { detail: 'Unexpected test request' } });
    }
    return route.fulfill({ json: body });
  });
  await page.goto(origin);
  await page.locator('#photoGrid .photo').first().waitFor();
  return { page, searches, saved };
}

test('search controls fit one row; filters and saved searches preserve their meaning', async t => {
  const { page, searches, saved } = await openApp(t);
  const boxes = await Promise.all(['#searchInput', '#searchMode', '#searchForm button'].map(s => page.locator(s).boundingBox()));
  assert.ok(boxes.every(b => Math.abs(b.y - boxes[0].y) < 4));
  await page.getByRole('button', { name: 'Filters', exact: true }).click();
  await page.getByLabel('Favorites only', { exact: true }).check();
  await page.waitForFunction(() => document.querySelector('#filtersButton').textContent === 'Filters (1)');
  assert.equal(searches.at(-1).favorites_only, true);
  await page.getByRole('button', { name: 'Done', exact: true }).click();
  assert.equal(await page.locator('#filterSummary').innerText(), 'Favorites only');
  await page.getByRole('button', { name: 'Save / manage…' }).click();
  await page.getByLabel('Saved search name').fill('My favorites');
  await page.getByRole('button', { name: 'Save search', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('#savedSearchDialog').open);
  assert.equal(saved[0].request.favorites_only, true);
  await page.locator('#gridSize').evaluate(el => { el.value = '260'; el.dispatchEvent(new Event('input')); });
  await page.reload();
  await page.locator('#photoGrid .photo').first().waitFor();
  assert.equal(await page.locator('#gridSize').inputValue(), '260');
});

test('album grid and viewer navigate both ways across page boundaries', async t => {
  const { page } = await openApp(t);
  await page.getByRole('tab', { name: 'Albums', exact: true }).click();
  await page.locator('.album-card').click();
  await page.waitForFunction(() => document.querySelector('#albumPageLabel').textContent === 'PAGE 1 / 2');
  assert.equal(await page.locator('#albumPhotos .photo').count(), 48);
  await page.locator('#albumPhotos .photo').last().click();
  await page.locator('#modalNext').click();
  await page.waitForFunction(() => document.querySelector('#modalName').textContent === 'holiday-048.jpg');
  assert.equal(await page.locator('#albumPhotos .photo').count(), 7);
  await page.locator('#modalPrev').click();
  await page.waitForFunction(() => document.querySelector('#modalName').textContent === 'holiday-047.jpg');
  await page.keyboard.press('Escape');
  await page.locator('#albumNextPage').click();
  await page.waitForFunction(() => document.querySelector('#albumPageLabel').textContent === 'PAGE 2 / 2');
  await page.locator('#albumPhotos .photo').first().focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => document.querySelector('#modalName').textContent === 'holiday-048.jpg');
});

test('modal focus is contained, nested shortcuts close first, and focus returns', async t => {
  const { page } = await openApp(t);
  const first = page.locator('#photoGrid .photo').first();
  await first.click();
  assert.equal(await page.locator('#modalClose').evaluate(el => el === document.activeElement), true);
  assert.equal(await page.locator('.shell').evaluate(el => el.inert), true);
  for (let i = 0; i < 25; i++) {
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => Boolean(document.activeElement.closest('#photoModal'))), true);
  }
  await page.locator('#modalClose').focus();
  await page.keyboard.press('?');
  assert.equal(await page.locator('#shortcutClose').evaluate(el => el === document.activeElement), true);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#photoModal').isVisible(), true);
  await page.keyboard.press('Escape');
  assert.equal(await first.evaluate(el => el === document.activeElement), true);
  assert.equal(await page.locator('.shell').evaluate(el => el.inert), false);
});

test('selection survives album paging without using a range anchor from another page', async t => {
  const { page } = await openApp(t);
  await page.getByRole('tab', { name: 'Albums', exact: true }).click();
  await page.locator('.album-card').click();
  await page.locator('#albumPhotos .photo').first().waitFor();
  await page.locator('#albumPhotos .photo').last().click({ modifiers: ['Control'] });
  await page.locator('#albumNextPage').click();
  await page.waitForFunction(() => document.querySelector('#albumPageLabel').textContent === 'PAGE 2 / 2');
  await page.locator('#albumPhotos .photo').first().click({ modifiers: ['Shift'] });
  assert.equal(await page.locator('#selectionCount').innerText(), '2 SELECTED · IN ALBUM');
  assert.equal(await page.locator('#albumPhotos .photo.selected').count(), 1);
  assert.equal(await page.locator('#photoModal').isVisible(), false);
});

test('narrow screens have no horizontal overflow', async t => {
  const { page } = await openApp(t, { width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.getByRole('button', { name: 'Filters', exact: true }).click();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  if (process.env.PHOTOLIB_SCREENSHOTS) {
    mkdirSync(process.env.PHOTOLIB_SCREENSHOTS, { recursive: true });
    await page.screenshot({ path: join(process.env.PHOTOLIB_SCREENSHOTS, 'filters-mobile.png') });
    await page.getByRole('button', { name: 'Done', exact: true }).click();
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.screenshot({ path: join(process.env.PHOTOLIB_SCREENSHOTS, 'library-desktop.png') });
  }
});
