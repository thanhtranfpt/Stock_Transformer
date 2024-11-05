import logging
import os


def get_logger(name, log_file='logs.log', level=logging.INFO, mode='a'):
    """
    Hàm tạo logger
    
    :param name: Tên của logger
    :param log_file: Tên file log (mặc định là '/content/logs.log')
    :param level: Cấp độ log (mặc định là logging.INFO)
    :param mode: Chế độ ghi file ('a' để thêm log, 'w' để ghi đè, mặc định là 'a')
    """
    # Tạo thư mục chứa log nếu chưa tồn tại
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # Xóa các handler trước đó nếu có (tránh trùng lặp)
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level=level)

    # Tạo StreamHandler để log ra console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Tạo FileHandler để log vào file
    file_handler = logging.FileHandler(log_file, mode=mode, encoding='UTF-8')
    file_handler.setLevel(logging.INFO)

    # Tạo định dạng log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Thêm các handler vào logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    # Lấy logger từ file logger_config.py
    logger = get_logger(__name__)

    # Sử dụng logger
    logger.info("This is an info message.")
    logger.error("This is an error message.")