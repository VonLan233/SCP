"""
API错误处理器
"""

def handle_400_error(error):
    """处理400错误"""
    return {
        'success': False,
        'error': 'Bad Request',
        'message': str(error) or '请求参数错误'
    }, 400

def handle_401_error(error):
    """处理401错误"""
    return {
        'success': False,
        'error': 'Unauthorized',
        'message': str(error) or '未授权的请求'
    }, 401

def handle_403_error(error):
    """处理403错误"""
    return {
        'success': False,
        'error': 'Forbidden',
        'message': str(error) or '禁止访问此资源'
    }, 403

def handle_404_error(error):
    """处理404错误"""
    return {
        'success': False,
        'error': 'Not Found',
        'message': str(error) or '请求的资源不存在'
    }, 404

def handle_405_error(error):
    """处理405错误"""
    return {
        'success': False,
        'error': 'Method Not Allowed',
        'message': str(error) or '不允许的请求方法'
    }, 405

def handle_422_error(error):
    """处理422错误"""
    return {
        'success': False,
        'error': 'Unprocessable Entity',
        'message': str(error) or '无法处理的实体'
    }, 422

def handle_429_error(error):
    """处理429错误"""
    return {
        'success': False,
        'error': 'Too Many Requests',
        'message': str(error) or '请求过于频繁'
    }, 429

def handle_500_error(error):
    """处理500错误"""
    # 在生产环境中，应该记录详细错误信息，但向用户返回通用信息
    return {
        'success': False,
        'error': 'Internal Server Error',
        'message': '服务器内部错误，请稍后再试'
    }, 500

def handle_503_error(error):
    """处理503错误"""
    return {
        'success': False,
        'error': 'Service Unavailable',
        'message': str(error) or '服务暂时不可用，请稍后再试'
    }, 503

# 错误处理器映射
api_error_handlers = {
    400: handle_400_error,
    401: handle_401_error,
    403: handle_403_error,
    404: handle_404_error,
    405: handle_405_error,
    422: handle_422_error,
    429: handle_429_error,
    500: handle_500_error,
    503: handle_503_error
}