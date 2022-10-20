import logging
import time
from contextlib import contextmanager
from enum import Enum
from functools import partial

import aiohttp
import anyio
import asyncio
import pymorphy2
import pytest
from aiohttp import web
from async_timeout import timeout

from adapters import inosmi_ru
from adapters.exceptions import ArticleNotFound
from text_tools import split_by_words, calculate_jaundice_rate

logger = logging.getLogger(__name__)


FETCHING_TIMEOUT = 5
SPLITTING_TIMEOUT = 3
MAX_ARTICLES_PER_REQUEST = 10


class ProcessingStatus(Enum):
    OK = 'OK'
    FETCH_ERROR = 'FETCH_ERROR'
    PARSING_ERROR = 'PARSING_ERROR'
    TIMEOUT = 'TIMEOUT'


@contextmanager
def measure_time():
    start_time = time.monotonic()
    end_time = start_time
    yield lambda: end_time - start_time
    end_time = time.monotonic()


def get_charged_words():
    with open('charged_dict/negative_words.txt', encoding='utf8') as file:
        charged_words = file.read().splitlines()
    with open('charged_dict/positive_words.txt', encoding='utf8') as file:
        charged_words.extend(file.read().splitlines())
    return charged_words


async def fetch_url(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def process_article(session, morph, charged_words, url, results, fetching_timeout=5, splitting_timeout=3):
    jaundice_rate, words_count, splitting_time = None, None, None
    try:
        if 'inosmi.ru' not in url:
            raise ArticleNotFound
        async with timeout(fetching_timeout):
            html = await fetch_url(session, url)
        text = inosmi_ru.sanitize(html, plaintext=True)
        with measure_time() as t:
            async with timeout(splitting_timeout):
                splitted_text = await split_by_words(morph, text)
        splitting_time = t()
        jaundice_rate = calculate_jaundice_rate(splitted_text, charged_words)
        words_count = len(splitted_text)
        status = ProcessingStatus.OK
    except (aiohttp.InvalidURL, aiohttp.ClientResponseError):
        status = ProcessingStatus.FETCH_ERROR
    except ArticleNotFound:
        status = ProcessingStatus.PARSING_ERROR
    except asyncio.exceptions.TimeoutError:
        status = ProcessingStatus.TIMEOUT
    results.append({
        'url': url,
        'status': status.value,
        'score': jaundice_rate,
        'words_count': words_count,
        'splitting_time': splitting_time
    })


@pytest.fixture
def event_loop():
    yield asyncio.get_event_loop()


def pytest_sessionfinish(session, exitstatus):
    asyncio.get_event_loop().close()


@pytest.mark.asyncio
async def test_process_article():
    morph = pymorphy2.MorphAnalyzer()
    charged_words = get_charged_words()

    results = []
    url = 'https://lenta.ru/brief/2021/08/26/afg_terror/'
    async with aiohttp.ClientSession() as session:
        await process_article(session, morph, charged_words, url, results)
    assert results[0]['status'] == 'PARSING_ERROR'

    results = []
    url = 'https://inosmi.ru/nopage.html'
    async with aiohttp.ClientSession() as session:
        await process_article(session, morph, charged_words, url, results)
    assert results[0]['status'] == 'FETCH_ERROR'

    results = []
    url = 'https://inosmi.ru/20221015/ukraina-256831968.html'
    async with aiohttp.ClientSession() as session:
        await process_article(session, morph, charged_words, url, results)
    assert results[0]['status'] == 'OK'
    async with aiohttp.ClientSession() as session:
        await process_article(session, morph, charged_words, url, results, fetching_timeout=0)
    assert results[1]['status'] == 'TIMEOUT'
    async with aiohttp.ClientSession() as session:
        await process_article(session, morph, charged_words, url, results, splitting_timeout=0)
    assert results[2]['status'] == 'TIMEOUT'


async def handle_web_request(request, morph, charged_words):
    urls = request.query.get('urls')
    if not urls:
        return web.json_response({'error': 'urls are not specified'}, status=400)
    urls = urls.split(',')
    if len(urls) > MAX_ARTICLES_PER_REQUEST:
        return web.json_response({'error': 'too many urls in request, should be 10 or less'}, status=400)
    results = []
    async with aiohttp.ClientSession() as session:
        async with anyio.create_task_group() as tg:
            for url in urls:
                tg.start_soon(
                    process_article,
                    session,
                    morph,
                    charged_words,
                    url,
                    results,
                    fetching_timeout=FETCHING_TIMEOUT,
                    splitting_timeout=SPLITTING_TIMEOUT
                )
    for result in results:
        del result['splitting_time']
    return web.json_response(results)


def main():
    logging.basicConfig(format='{name} - {levelname} - {message}', style='{', level=logging.INFO)

    morph = pymorphy2.MorphAnalyzer()
    charged_words = get_charged_words()

    handle_web_request_partial = partial(handle_web_request, morph=morph, charged_words=charged_words)
    app = web.Application()
    app.add_routes([web.get('/', handle_web_request_partial),
                    web.get('/{name}', handle_web_request_partial)])
    web.run_app(app)


if __name__ == '__main__':
    main()
