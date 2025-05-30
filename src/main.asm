org 0x7C00
bits 16

%define ENDL 0x0D, 0x0A

start:
    jmp main

; Prints a string on the screen
; Params:
;       - ds:si points to string
puts:
    ; save registers we will modify
    push si
    push ax

.loop:
    lodsb               ; load next character into AL
    or al, al           ; check if null terminator
    jz .done

    mov ah, 0x0E        ; teletype BIOS service
    int 0x10

    jmp .loop

.done:
    pop ax
    pop si
    ret

main:
    ; setup data segments
    mov ax, 0
    mov ds, ax
    mov es, ax

    ; setup stack
    mov ss, ax
    mov sp, 0x7C00      ; stack just below bootloader

    ; print message
    mov si, msg_hello
    call puts           ; use 'call', not 'tail'

    hlt

.halt:
    jmp .halt

msg_hello: db 'Welcome!', ENDL, 0

times 510 - ($ - $$) db 0
dw 0xAA55

