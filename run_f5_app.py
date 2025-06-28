# run_cli.py
import logging
import click
from . import config
from F5_TTS_RU.f5_tts_ru_app import create_gradio_app

@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Port to run the app on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to run the app on")
@click.option("--share", "-s", is_flag=True, default=False, help="Share via Gradio link")
@click.option("--api", "-a", is_flag=True, default=True, help="Allow API access")
@click.option("--gpu", type=int, default=config.CURRENT_GPU, help="Which GPU to use (e.g. 0 or 1)")
def main(port, host, share, api, gpu):
    logging.info(f"🔧 Using GPU {gpu} on port {port}")
    config.CURRENT_GPU = gpu

    app = create_gradio_app()
    app.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        debug=True
    )

if __name__ == "__main__":
    main()
