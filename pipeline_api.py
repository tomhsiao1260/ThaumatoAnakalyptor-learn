import sys
import requests
from flask import Flask, jsonify


class Node:
    def __init__(self, handler):
        self.app = Flask(__name__)
        self.handler = handler
        self.current_id, self.from_id = self.get_flow_info()
        self.setup_routes()
        self.run()

    def get_flow_info(self):
        args = sys.argv[1:]

        current_id = None
        from_id = None

        for arg in args:
            if arg.startswith("id="):
                current_id = arg.split("=")[1]
            elif arg.startswith("from="):
                from_id = arg.split("=")[1]

        return current_id, from_id

    def get_inputs(self):
        if self.from_id == "5000":
            return {
                "data": {},
                "view": {},
            }
        else:
            inputs = requests.get("http://127.0.0.1:" + self.from_id).json()
            return inputs

    def process(self):
        inputs = self.get_inputs()

        outputs = self.handler(inputs)

        return jsonify(outputs)

    def setup_routes(self):
        @self.app.route("/")
        # DO NOT DELETE "process_route" function
        def process_route():
            return self.process()

    def run(self):
        self.app.run(port=self.current_id)