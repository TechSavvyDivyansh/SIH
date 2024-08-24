import express from 'express'
import chatCotroller from '../controller/chat_controller.js'

const router=express.Router()

router.post('/savechat',chatCotroller.saveChat)
router.get('/get-initial-chat',chatCotroller.getInitalChats)

export default router