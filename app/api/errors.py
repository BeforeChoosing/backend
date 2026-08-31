"""Stable public error contract; keep detail for older frontend clients."""
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.services.request_context import get_request_context
from app.services.runtime_log import log_event

ERRORS = {
    400: ('INVALID_REQUEST', '请求内容有误，请检查后重试。'),
    401: ('AUTH_REQUIRED', '请登录后再继续。'),
    403: ('ACCESS_DENIED', '你暂时没有权限执行此操作。'),
    404: ('NOT_FOUND', '内容不存在或已被移除，请刷新后重试。'),
    409: ('STATE_CONFLICT', '内容状态已变化，请刷新后重试。'),
    413: ('PAYLOAD_TOO_LARGE', '材料过大，请将单个文件控制在 20MB 以内。'),
    422: ('VALIDATION_ERROR', '提交内容不完整或格式有误，请检查后重试。'),
    429: ('RATE_LIMITED', '操作有些频繁，请稍等片刻再试。'),
    499: ('CLIENT_CANCELLED', '请求已取消。'),
    500: ('INTERNAL_ERROR', '服务暂时出现问题，请稍后重试。'),
    502: ('UPSTREAM_ERROR', '这次 AI 回复未能完成，请稍后重试。'),
    503: ('SERVICE_UNAVAILABLE', '服务暂时不可用，请稍后重试。'),
    504: ('UPSTREAM_TIMEOUT', '这次处理时间较长，请稍后查看结果，再决定是否重试。'),
}


def error_response(status: int, detail=None, headers=None):
    code, message = ERRORS.get(status, ('REQUEST_FAILED', '操作未完成，请稍后重试。'))
    # Only expected business errors may carry application-authored instructions.
    if status < 500 and isinstance(detail, str) and len(detail) <= 300:
        message = detail
    request_id = get_request_context().request_id
    return JSONResponse(status_code=status, headers=headers, content={
        'detail': message, 'error': {'code': code, 'message': message,
                                    'request_id': request_id, 'retryable': status in {502, 503, 504}},
        'request_id': request_id,
    })


async def http_error_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        log_event('request_failed', level='error', error=exc.__cause__ or exc, alert=True,
                  status_code=exc.status_code, error_code=ERRORS.get(exc.status_code, ('REQUEST_FAILED',))[0])
    return error_response(exc.status_code, exc.detail, exc.headers)


async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Pydantic errors can echo passwords and answers in input/ctx; don't return them.
    return error_response(422)
