import asyncio
from random import shuffle
import random
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
# from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


VALID_TIME_FOR_ROOM = 600


app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
room_data = {}


async def clean_outdated_room(interval: int):
    while True:
        outdated = []
        for k in room_data:
            if room_data[k]['time'] < time.time():
                outdated.append(k)
        for k in outdated:
            del room_data[k]
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup_event():
    _ = asyncio.create_task(clean_outdated_room(30))


@app.get("/")
def read_root():
    return FileResponse('frontend/dist/index.html')


class CreateRoomRequest(BaseModel):
    room_id: str
    command: str
    charactor_num: int


@app.post("/create")
def handle_command(req: CreateRoomRequest):
    """
    Create a new room. If room not exist, create it. otherwise, notify it 
    exists.
    """
    try:
        rid = req.room_id
        if rid in room_data:
            return JSONResponse({"status": "exist"})
        else:
            cmd = req.command.strip().split(' ')
            init = cmd[0]
            cmd = cmd[1:]
            if random.random() < 0.5:
                # reverse 0p and 1p commands
                cmd = [f'{1 - int(x[0])}{x[1]}' for x in cmd]
            if init[0] != 'i':
                return HTTPException(status_code=400, detail="first not init")
                # return JSONResponse({"status": "error"})
            init = int(init[1:])
            bp_list = list(range(req.charactor_num))
            shuffle(bp_list)
            bp_list = bp_list[:init]
            bp_list.sort()
            room_data[rid] = {
                'done_command': [],
                'next_command': cmd,
                'charactor_num': req.charactor_num,
                'time': time.time() + VALID_TIME_FOR_ROOM,
                'bp_list': bp_list,
                'bp_status': [''] * init,
                'players': [
                    [], []
                ],
                'results': ['', '']
            }
            return JSONResponse({"status": "ok"})
    except Exception as e:
        raise e
        # return JSONResponse({"status": "error"})


@app.delete("/room/{room_id}")
def handle_delete_room(room_id: str):
    """
    Delete room. If room not exist, return error.
    """
    if room_id in room_data:
        del room_data[room_id]
        return JSONResponse({"status": "ok"})
    else:
        return HTTPException(status_code=404, detail="room not found")


@app.get("/room/{room_id}")
def handle_room(room_id: str):
    """
    Get room data. If room not exist, return error.
    """
    if room_id in room_data:
        return JSONResponse(room_data[room_id])
    else:
        return HTTPException(status_code=404, detail="room not found")


class ActData(BaseModel):
    player_id: int
    index: int


@app.post("/room/{room_id}/act")
def handle_act(room_id: str, act_data: ActData):
    """
    Handle act command. If room not exist, return error. If act is invalid,
    return error. Otherwise, return room_data.
    """
    if room_id in room_data:
        data = room_data[room_id]
        if len(data['next_command']) == 0:
            return HTTPException(status_code=400, detail="no command")
        cmd = data['next_command'][0]
        raw_cmd = cmd
        if int(cmd[0]) != act_data.player_id:
            return HTTPException(status_code=400, detail="wrong player")
        if data['bp_status'][act_data.index] != '':
            return HTTPException(status_code=400, detail="used charactor")
        cmd = cmd[1]
        if cmd == 'b':
            cmd = 'ban'
        elif cmd == 'p':
            cmd = 'pick'
        else:
            return HTTPException(status_code=400, detail="wrong command")
        data['bp_status'][act_data.index] = cmd
        data['next_command'] = data['next_command'][1:]
        data['done_command'].append(raw_cmd)
        if cmd == 'pick':
            data['players'][act_data.player_id].append(
                data['bp_list'][act_data.index])
        return data
    else:
        return HTTPException(status_code=404, detail="room not found")


class ResultData(BaseModel):
    player_id: int
    result: str


@app.post("/room/{room_id}/result")
def handle_result(room_id: str, res: ResultData):
    """
    Handle result command. If room not exist, return error. Otherwise, return
    room_data.
    """
    if room_id in room_data:
        data = room_data[room_id]
        if len(data['next_command']) != 0:
            return HTTPException(status_code=400, detail="command not done")
        data['results'][res.player_id] = res.result
        return data
    else:
        return HTTPException(status_code=404, detail="room not found")


app.mount("/", StaticFiles(directory="frontend/dist"), name="")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
